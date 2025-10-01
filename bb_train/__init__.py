# bb_train: Training pipeline for builderbrain
# Coordinates model training with dual constraints and multi-objective optimization

from . import trainer
from . import data_loader
from . import config

__all__ = ["trainer", "data_loader", "config"]
