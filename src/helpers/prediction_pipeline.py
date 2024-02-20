from src.dataloader.data_collection import DataCollection
from src.feature.feature_generation import FeatureExtractor
import logging
import pandas as pd
from typing import Tuple
import copy
from src.helpers.training_pipeline import prepare_model_data


def prediction_data_pipeline(path: str = None, skip_feature_generator: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Collects data and optionally generates features, returning a DataFrame suitable for model prediction.

    Args:
        path (str): Path to the dataset to be processed.
        skip_feature_generation (bool): Flag to skip feature generation if set to True.

    Returns:
        pd.DataFrame: The DataFrame containing processed features for prediction.
    """

    try:

        if skip_feature_generator:

            logging.info("Skipping Feature Generation, The provided file is already processed")

            df = pd.read_csv(path)
        else:
            data_collector = DataCollection(path)
            logging.info("Data Collector initialized.")
            dataframe = data_collector.get_dataframe()
            logging.info("restaurants_ids generated.")
            feature_generator = FeatureExtractor(copy.deepcopy(dataframe), data_collector.restaurants_ids)
            logging.info("Feature_generator initialized. Generating features with feature_generator.")
            feature_generator.generate_features()
            logging.info("Generating features successful.")
            df = feature_generator.df

        return prepare_model_data(df)

    except Exception as e:
        logging.error(f"Error in prediction data pipeline: {e}")
        raise
