"""This module contains all the constants that is used along training universe
"""
from enum import Enum
import os


class EnviromentVariables(Enum):
    """
    Enviromental variables for all repo constants
    """
    BUCKET_NAME = "project-main"
    TRAINING_UNIVERSE_REMOTE_SUFFIX = "training-universe/"
    MODEL_REGISTRY_REMOTE_SUFFIX = "s3://project-main/training-universe/api-models/"
    TRAINING_UNIVERSE_DVC_CACHE = "s3://project-main/training-universe/dvc-cache/"
    BASE_EXPERIMENTS_DIR = os.path.abspath("exp")


class LoggerTypes(Enum):
    """
    Constant names of logger types
    """
    NEPTUNE = "neptune"
    LOCAL = "local"




class HyperparameterKeys(Enum):
    """
    Constant names of hyperparameters that can be tuned
    """

