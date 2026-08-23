"""Evaluate UCS-Expert and optionally save binary prediction masks."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CoralDataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from UCSExpert import UCSExpert
from utils.checkpoint import load_model_checkpoint
from utils.metrics import (
    dice_coefficient,
    intersection_over_union,
    mean_absolute_error,
    pixel_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type",
                        default="vit_b",
                        choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--checkpoint", default="sam_ckp/sam_vit_b_01ec64.pth")
    parser.add_argument("--resume", default="checkpoint/ucs_b.pth")
    parser.add_argument("--data_path", default="sample")
    parser.add_argument("--save_dir", default="output")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--label_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--save_pic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = CoralDataLoader(
        args.data_path,
        train=False,
        resize_size=(args.image_size, args.image_size),
        label_resize_size=(args.label_size, args.label_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    sam = sam_model_registry[args.model_type](image_size=args.image_size,
                                              checkpoint=args.checkpoint)
    model = UCSExpert(sam, vit_type=args.model_type)
    load_model_checkpoint(model, args.resume)
    model.to(device).eval()
    box_transform = ResizeLongestSide(model.sam.image_encoder.img_size)

    save_dir = Path(args.save_dir)
    if args.save_pic:
        save_dir.mkdir(parents=True, exist_ok=True)

    totals = {"dice": 0.0, "iou": 0.0, "accuracy": 0.0, "mae": 0.0}
    with torch.inference_mode():
        progress = tqdm(dataloader, desc="Evaluating")
        for image, target, box, _, image_paths, _ in progress:
            transformed_box = box_transform.apply_boxes(
                box.numpy(), (target.shape[-2], target.shape[-1]))
            box_tensor = torch.as_tensor(transformed_box[:, None, :],
                                         dtype=torch.float32,
                                         device=device)
            image = image.to(device, non_blocking=True)

            amp_enabled = args.use_amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(image,
                                    box_tensor,
                                    original_size=target.shape[-2:])
                logits = F.interpolate(
                    predictions[-1],
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            prediction = (torch.sigmoid(logits)
                          > 0.5).cpu().numpy().astype(np.uint8)
            ground_truth = target.numpy().astype(np.uint8)
            sample_metrics = {
                "dice": dice_coefficient(ground_truth, prediction),
                "iou": intersection_over_union(ground_truth, prediction),
                "accuracy": pixel_accuracy(ground_truth, prediction),
                "mae": mean_absolute_error(ground_truth, prediction),
            }
            for name, value in sample_metrics.items():
                totals[name] += float(value)
            progress.set_postfix({
                name: f"{value:.4f}"
                for name, value in sample_metrics.items()
            })

            if args.save_pic:
                original = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
                if original is None:
                    raise RuntimeError(
                        f"Unable to read source image: {image_paths[0]}")
                height, width = original.shape[:2]
                restored = cv2.resize(prediction[0, 0], (width, height),
                                      interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(save_dir / f"{Path(image_paths[0]).stem}.png"),
                            restored * 255)

    count = len(dataset)
    summary = " | ".join(f"{name}: {value / count:.5f}"
                         for name, value in totals.items())
    print(f"Evaluated {count} images | {summary}")
    if args.save_pic:
        print(f"Predictions saved to: {save_dir.resolve()}")


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu instead."
        )
    return torch.device(requested)


if __name__ == "__main__":
    main()
