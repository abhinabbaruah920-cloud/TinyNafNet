"""Central configuration defaults."""
from dataclasses import dataclass

@dataclass(frozen=True)
class TrainConfig:
    width: int = 24
    middle_blocks: int = 2
    scale: int = 2
    patch_size: int = 256
    batch_size: int = 16
    lr: float = 2e-4
    epochs: int = 200
    val_split: float = 0.1
