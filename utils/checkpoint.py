"""Checkpoint loading helpers shared by training, evaluation, and the GUI."""

from pathlib import Path
from typing import Any, Dict, Union

import torch


def load_checkpoint(
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path,
                      map_location=map_location,
                      weights_only=True)


def load_model_checkpoint(model: torch.nn.Module,
                          path: Union[str, Path]) -> Dict[str, Any]:
    checkpoint = load_checkpoint(path)
    state_dict = checkpoint.get("model", checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    return checkpoint
