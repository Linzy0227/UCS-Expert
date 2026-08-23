"""Interactive bounding-box GUI for UCS-Expert."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from segment_anything import sam_model_registry
from UCSExpert import UCSExpert
from utils.checkpoint import load_model_checkpoint

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type",
                        default="vit_b",
                        choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--checkpoint", default="sam_ckp/sam_vit_b_01ec64.pth")
    parser.add_argument("--resume", default="checkpoint/ucs_b.pth")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(args: argparse.Namespace, device: torch.device) -> UCSExpert:
    print("Loading UCS-Expert model...")
    start = time.perf_counter()
    sam = sam_model_registry[args.model_type](image_size=args.image_size,
                                              checkpoint=args.checkpoint)
    model = UCSExpert(sam, vit_type=args.model_type)
    load_model_checkpoint(model, args.resume)
    model.to(device).eval()
    print(
        f"Model loaded in {time.perf_counter() - start:.2f} seconds on {device}."
    )
    return model


@torch.inference_mode()
def predict_mask(
    model: UCSExpert,
    image: torch.Tensor,
    box: np.ndarray,
    original_size,
) -> np.ndarray:
    box_tensor = torch.as_tensor(box[:, None, :],
                                 dtype=torch.float32,
                                 device=image.device)
    prediction = model(image, box_tensor, original_size=original_size)[-1]
    prediction = F.interpolate(
        torch.sigmoid(prediction),
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )
    return (prediction[0, 0].cpu().numpy() > 0.5).astype(np.uint8)


def array_to_pixmap(image: np.ndarray) -> QPixmap:
    image = np.ascontiguousarray(image.astype(np.uint8))
    height, width, _ = image.shape
    qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class Window(QWidget):

    def __init__(self, model: UCSExpert, device: torch.device,
                 image_size: int) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.image_size = image_size
        self.half_point_size = 5
        self.point_size = self.half_point_size * 2

        self.image_path = None
        self.image = None
        self.image_tensor = None
        self.mask = None
        self.previous_mask = None
        self.color_index = 0
        self.is_mouse_down = False
        self.start_position = None
        self.start_point = None
        self.end_point = None
        self.rectangle = None
        self.background_item = None

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")
        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)

        button_layout = QHBoxLayout()
        button_layout.addWidget(load_button)
        button_layout.addWidget(save_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        layout.addLayout(button_layout)

        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(self.close)
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        self.setWindowTitle("UCS-Expert")

    def load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Image to Segment",
            ".",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return

        self.image_path = Path(file_path)
        with Image.open(file_path) as source_image:
            self.image = np.asarray(source_image.convert("RGB"))
        resized = cv2.resize(
            self.image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        self.image_tensor = (torch.from_numpy(resized.copy()).float().permute(
            2, 0, 1).unsqueeze(0).div(255.0).to(self.device))

        height, width = self.image.shape[:2]
        self.scene = QGraphicsScene(0, 0, width, height)
        self.background_item = self.scene.addPixmap(array_to_pixmap(
            self.image))
        self.mask = np.zeros((height, width, 3), dtype=np.uint8)
        self.previous_mask = None
        self.color_index = 0
        self.end_point = None
        self.rectangle = None
        self.view.setScene(self.scene)
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

    def mouse_press(self, event) -> None:
        if self.image is None:
            return
        x, y = event.scenePos().x(), event.scenePos().y()
        self.is_mouse_down = True
        self.start_position = (x, y)
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
        )

    def mouse_move(self, event) -> None:
        if not self.is_mouse_down:
            return
        x, y = event.scenePos().x(), event.scenePos().y()
        if self.end_point is not None:
            self.scene.removeItem(self.end_point)
        self.end_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
        )
        if self.rectangle is not None:
            self.scene.removeItem(self.rectangle)
        start_x, start_y = self.start_position
        x_min, x_max = sorted((x, start_x))
        y_min, y_max = sorted((y, start_y))
        self.rectangle = self.scene.addRect(x_min,
                                            y_min,
                                            x_max - x_min,
                                            y_max - y_min,
                                            pen=QPen(QColor("red")))

    def mouse_release(self, event) -> None:
        if not self.is_mouse_down:
            return
        self.is_mouse_down = False
        x, y = event.scenePos().x(), event.scenePos().y()
        start_x, start_y = self.start_position
        height, width = self.image.shape[:2]
        x_min, x_max = np.clip(sorted((x, start_x)), 0, width - 1)
        y_min, y_max = np.clip(sorted((y, start_y)), 0, height - 1)
        if x_max - x_min < 2 or y_max - y_min < 2:
            return

        scale = np.array([
            self.image_size / width,
            self.image_size / height,
            self.image_size / width,
            self.image_size / height,
        ])
        box = np.array([[x_min, y_min, x_max, y_max]],
                       dtype=np.float32) * scale
        binary_mask = predict_mask(self.model, self.image_tensor, box,
                                   (height, width))
        self.previous_mask = self.mask.copy()
        self.mask[binary_mask > 0] = COLORS[self.color_index % len(COLORS)]
        self.color_index += 1
        self.refresh_overlay()

    def refresh_overlay(self) -> None:
        overlay = self.image.copy()
        selected = np.any(self.mask != 0, axis=2)
        overlay[selected] = (0.8 * overlay[selected] +
                             0.2 * self.mask[selected]).astype(np.uint8)
        self.scene.removeItem(self.background_item)
        self.background_item = self.scene.addPixmap(array_to_pixmap(overlay))
        self.background_item.setZValue(-1)

    def undo(self) -> None:
        if self.previous_mask is None:
            print("No previous mask to restore.")
            return
        self.mask = self.previous_mask
        self.previous_mask = None
        self.color_index = max(0, self.color_index - 1)
        self.refresh_overlay()

    def save_mask(self) -> None:
        if self.image_path is None:
            print("No image loaded.")
            return
        output_path = self.image_path.with_name(
            f"{self.image_path.stem}_mask.png")
        Image.fromarray(self.mask).save(output_path)
        print(f"Mask saved to: {output_path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = build_model(args, device)
    application = QApplication(sys.argv)
    window = Window(model, device, args.image_size)
    window.resize(1000, 700)
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
