import neptune
from src.configs.enums import LoggerTypes


def get_logger(configs):
    if configs.logger.type == LoggerTypes.NEPTUNE.value:

        logger = neptune.init_run(
            project=configs.logger.settings.project_name,
            name=configs.logger.settings.experiment_name,
            tags=configs.logger.settings.tags,
        )
        return logger
    elif configs.logger.type == LoggerTypes.LOCAL:
        pass
