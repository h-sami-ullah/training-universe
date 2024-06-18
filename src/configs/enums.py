"""This module contains all the constants that is used along training universe
"""

from enum import Enum
import os


class EnvironmentVariables(Enum):
    """
    Environment variables defining key constants for the project.

    Attributes:
        BUCKET_NAME (str): The name of the S3 bucket used for storing project data.
        TRAINING_UNIVERSE_REMOTE_SUFFIX (str): The remote directory suffix for training universe-related data.
        MODEL_REGISTRY_REMOTE_SUFFIX (str): The remote directory suffix for API models artifacts.
        TRAINING_UNIVERSE_DVC_CACHE (str): The S3 URI for DVC cache storage.
    """

    BUCKET_NAME = "project-main"
    TRAINING_UNIVERSE_REMOTE_SUFFIX = "training-universe/"
    MODEL_REGISTRY_REMOTE_SUFFIX = "training-universe/api-models/"
    TRAINING_UNIVERSE_DVC_CACHE = "s3://project-main/training-universe/dvc-cache/"


class LoggerTypes(Enum):
    """
    Defines constants for different types of loggers used in the project.

    Attributes:
        NEPTUNE (str): Identifier for the Neptune logger.
        LOCAL (str): Identifier for a local file-based logger.
    """

    NEPTUNE = "neptune"
    LOCAL = "local"
