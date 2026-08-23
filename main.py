"""Train UCS-Expert on a paired image/mask dataset."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import monai
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import CoralDataLoader
from dist import all_reduce_average, all_reduce_sum, init_distributed_mode, is_primary
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from UCSExpert import UCSExpert
from utils.checkpoint import load_checkpoint
from utils.metrics import dice_coefficient, intersection_over_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_type",
                        default="vit_b",
                        choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--task_name", default="UCS-Expert")
    parser.add_argument("--checkpoint", default="sam_ckp/sam_vit_b_01ec64.pth")
    parser.add_argument("--data_path", default="dataset/CoralMask")
    parser.add_argument("--work_dir", default="work_dir")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--label_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--dist-url", default="env://")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(2023)
    if args.distributed:
        init_distributed_mode(args)
    device = get_device(args)

    output_dir = Path(args.work_dir) / args.task_name
    if is_primary():
        output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CoralDataLoader(
        args.data_path,
        train=True,
        resize_size=(args.image_size, args.image_size),
        label_resize_size=(args.label_size, args.label_size),
        train_ratio=args.train_ratio,
    )
    validation_dataset = CoralDataLoader(
        args.data_path,
        train=False,
        resize_size=(args.image_size, args.image_size),
        label_resize_size=(args.label_size, args.label_size),
    )
    train_sampler = DistributedSampler(
        train_dataset, shuffle=True) if args.distributed else None
    validation_sampler = (DistributedSampler(validation_dataset, shuffle=False)
                          if args.distributed else None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if is_primary():
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(validation_dataset)}")

    sam = sam_model_registry[args.model_type](image_size=args.image_size,
                                              checkpoint=args.checkpoint)
    model = UCSExpert(sam, vit_type=args.model_type).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                  start_factor=1.0,
                                                  end_factor=0.001,
                                                  total_iters=1000)
    criterion = monai.losses.DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction="mean",
        lambda_dice=0.8,
        lambda_ce=0.2,
    )

    start_epoch = restore_training_state(model, optimizer, args.resume)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True,
        )

    box_transform = ResizeLongestSide(args.image_size)
    amp_enabled = args.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_history: List[float] = []
    dice_history: List[float] = []
    iou_history: List[float] = []
    best_dice = -1.0

    for epoch in range(start_epoch, args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            box_transform,
            device,
            amp_enabled,
            epoch,
            args.num_epochs,
        )
        scheduler.step()
        loss_history.append(epoch_loss)

        if is_primary():
            print(
                f"Epoch {epoch + 1:03d}/{args.num_epochs:03d} | loss: {epoch_loss:.6f}"
            )

        model_state = unwrap_model(model).state_dict()
        save_checkpoint(output_dir / "model_latest.pth", model_state,
                        optimizer, epoch, epoch_loss)
        if is_primary():
            plot_history(output_dir / "train_loss.png", [loss_history],
                         ["Loss"])

        if (epoch + 1) % args.eval_step != 0:
            continue

        metrics = evaluate(model, validation_loader, box_transform, device,
                           amp_enabled)
        dice_history.append(metrics["dice"])
        iou_history.append(metrics["iou"])
        if is_primary():
            print(
                f"Validation | Dice: {metrics['dice']:.5f} | IoU: {metrics['iou']:.5f}"
            )
            plot_history(
                output_dir / "validation_metrics.png",
                [dice_history, iou_history],
                ["Dice", "IoU"],
            )

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            save_checkpoint(
                output_dir / "model_best.pth",
                model_state,
                optimizer,
                epoch,
                epoch_loss,
                metrics,
            )

    if is_primary():
        print(f"Training complete. Outputs saved to: {output_dir.resolve()}")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: torch.amp.GradScaler,
    box_transform: ResizeLongestSide,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
    num_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    progress: Iterable = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        disable=not is_primary(),
    )
    for image, target, box, _, _, _ in progress:
        batch_size = image.shape[0]
        box_tensor = transform_boxes(box, target, box_transform, device)
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            predictions = model(image,
                                box_tensor,
                                original_size=target.shape[-2:])
            loss = sum(
                criterion(resize_prediction(prediction, target), target) /
                batch_size for prediction in predictions)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        reduced_loss = all_reduce_average(
            loss.detach()) if is_distributed(model) else loss.detach()
        total_loss += reduced_loss.item()
        if is_primary():
            progress.set_postfix(loss=f"{reduced_loss.item():.5f}")

    return total_loss / len(dataloader)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    box_transform: ResizeLongestSide,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    progress: Iterable = tqdm(dataloader,
                              desc="Validation",
                              disable=not is_primary())
    for image, target, box, _, _, _ in progress:
        box_tensor = transform_boxes(box, target, box_transform, device)
        image = image.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            predictions = model(image,
                                box_tensor,
                                original_size=target.shape[-2:])
        prediction = (torch.sigmoid(resize_prediction(predictions[-1], target))
                      > 0.5).cpu().numpy().astype(np.uint8)
        ground_truth = target.numpy().astype(np.uint8)
        totals[0] += dice_coefficient(ground_truth, prediction)
        totals[1] += intersection_over_union(ground_truth, prediction)
        totals[2] += image.shape[0]

    totals = all_reduce_sum(totals)
    return {
        "dice": (totals[0] / totals[2]).item(),
        "iou": (totals[1] / totals[2]).item(),
    }


def transform_boxes(
    boxes: torch.Tensor,
    targets: torch.Tensor,
    transform: ResizeLongestSide,
    device: torch.device,
) -> torch.Tensor:
    transformed = transform.apply_boxes(boxes.numpy(),
                                        (targets.shape[-2], targets.shape[-1]))
    return torch.as_tensor(transformed[:, None, :],
                           dtype=torch.float32,
                           device=device)


def resize_prediction(prediction: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction
    return F.interpolate(prediction,
                         size=target.shape[-2:],
                         mode="bilinear",
                         align_corners=False)


def build_optimizer(model: UCSExpert, learning_rate: float,
                    weight_decay: float) -> torch.optim.Optimizer:
    fine_decoder = list(model.fine_decoder.parameters())
    fine_decoder_ids = {id(parameter) for parameter in fine_decoder}
    other_trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in fine_decoder_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": fine_decoder,
                "lr": learning_rate
            },
            {
                "params": other_trainable,
                "lr": 0.05 * learning_rate
            },
        ],
        weight_decay=weight_decay,
    )


def restore_training_state(model: UCSExpert, optimizer: torch.optim.Optimizer,
                           resume_path: str) -> int:
    if not resume_path:
        return 0
    checkpoint = load_checkpoint(resume_path)
    state_dict = checkpoint.get("model", checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    if "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:
            print(
                "Warning: optimizer layout differs; restored model weights only."
            )
    return int(checkpoint.get("epoch", -1)) + 1


def save_checkpoint(
    path: Path,
    model_state: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    if not is_primary():
        return
    payload = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, path)


def plot_history(path: Path, series: List[List[float]],
                 labels: List[str]) -> None:
    import matplotlib.pyplot as plt

    for values, label in zip(series, labels):
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    if len(labels) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model,
                                      DistributedDataParallel) else model


def is_distributed(model: torch.nn.Module) -> bool:
    return isinstance(model, DistributedDataParallel)


def get_device(args: argparse.Namespace) -> torch.device:
    if args.distributed:
        return torch.device("cuda", args.gpu)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Use --device cpu instead."
        )
    return torch.device(args.device)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
