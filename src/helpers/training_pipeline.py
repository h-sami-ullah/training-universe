from src.data_collection import DataCollection
from src.feature_generation import FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.model_selection import GridSearchCV
from src.utils.dvc_controller import DvcDataSaver
import glob
from src.utils.s3_handler import S3handler
from src.configs.enums import EnvironmentVariables
import logging
import pandas as pd
from typing import Tuple, Dict, Any, Union
from sklearn.base import BaseEstimator
import copy
import time


def training_data_pipeline(configs: Dict[str, Any],
                           skip_feature_generator: bool = False) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Collect data, generate features, and split into training and testing datasets.

    Args:
        configs (Dict[str, Any]): Configuration dictionary containing settings for data collection,
        feature generation, and model training.
        skip_feature_generator (bool): Flag to skip feature generation if set to True.
        predict_flag (bool): Flag to indicate if data is used for prediction.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    """

    try:
        path_to_save = os.path.join(os.getcwd(), configs['data_management']['save_folder_processed'], 'processed.csv')

        if skip_feature_generator:
            logging.info("Skipping Feature Generation, if already exist")
            if os.path.isfile(path_to_save):

                logging.info("Feature already extracted.")
                logging.info(f"Reading Processed file from {path_to_save}")
                df = pd.read_csv(path_to_save)
            else:
                error_msg = "Cannot skip feature generation: File does not exist."
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)
        else:
            data_collector = DataCollection(**configs['data_collection_settings'])
            logging.info("Data Collector initialized.")
            dataframe = data_collector.get_dataframe()
            logging.info("restaurants_ids generated.")
            feature_generator = FeatureExtractor(copy.deepcopy(dataframe), data_collector.restaurants_ids)
            logging.info("Feature_generator initialized. Generating features with feature_generator.")
            feature_generator.generate_features()
            logging.info("Generating features successful.")

            # Save original and processed dataframes
            data_collector.save_dataframe(os.path.join(configs['data_management']['save_folder_processed'], 'original'
                                                                                                            '.csv'))
            feature_generator.save_dataframe(os.path.join(configs['data_management']['save_folder_processed'], 'proces'
                                                                                                               'sed.csv'
                                                          ))

            # Prepare data for model training
            df = feature_generator.df

        X, y = prepare_model_data(df)

        return split_data(X, y, configs['data_management']['train_test_split'])

    except Exception as e:
        logging.error(f"Error in training data pipeline: {e}")
        raise


def split_data(X: pd.DataFrame, y: pd.Series, split_config: Dict[str, Any]) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target series.
        split_config (Dict[str, Any]): Configuration for train-test split including test size and random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Split training and testing features and targets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_config['test_size'], random_state=split_config['random_seed'])
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare model data by selecting features and target.

    Args:
        df (pd.DataFrame): The DataFrame containing all data.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features DataFrame and target series.
    """
    # Assuming 'df' contains the columns listed below as features, adjust as needed
    feature_columns = [
        'dist_to_restaurant', 'Hdist_to_restaurant', 'avg_Hdist_to_restaurants',
        'date_day_number', 'restaurant_id', 'Five_Clusters_embedding', 'h3_index',
        'date_hour_number', 'restaurants_per_index'
    ]

    # Assuming 'orders_busyness_by_h3_hour' is the target variable
    target_column = 'orders_busyness_by_h3_hour'

    # Selecting the specified columns from the DataFrame
    X = df[feature_columns]

    # Assuming the target variable is a single column in the DataFrame
    y = df[target_column]

    return X, y


def get_model(configs: Dict[str, Any]) -> BaseEstimator:
    """
    Configure and return a machine learning model based on the provided configs.

    Args:
        configs (Dict[str, Any]): Configuration dictionary containing model settings.

    Returns:
        BaseEstimator: Configured model or GridSearchCV object if parameter search is enabled.
    """
    try:
        model_type = configs['model']['model_type']
        model = None

        # Initialize the model based on the model_type
        if model_type == "RandomForestRegressor":
            model = RandomForestRegressor(random_state=0, n_jobs=-1)
        # Add more model types here as elif statements
        # elif model_type == "AnotherModel":
        #     model = AnotherModel(...)

        if model is None:
            raise ValueError(f"Model type '{model_type}' is not supported.")

        # Check if parameter search is enabled
        if configs['model']['parameter_search_enabled']:
            # Ensure grid_search parameters are properly structured for GridSearchCV
            grid_search_params = {
                'param_grid': configs['model']['params_for_grid_search'],
                'cv': configs['model']['grid_search']['cv'],
                'n_jobs': configs['model']['grid_search']['n_jobs'],
                'verbose': configs['model']['grid_search']['verbose'],
                'scoring': configs['model']['grid_search']['scoring']
            }
            grid_search = GridSearchCV(estimator=model, **grid_search_params)
            return grid_search
        else:
            return model
    except Exception as e:
        logging.error(f"Error configuring the model: {e}")
        raise


def dvc_handler(configs: Dict[str, Any]):
    """
    Handle DVC operations based on configs.

    Args:
        configs (Dict[str, Any]): Configuration dictionary specifying DVC settings.
    """
    if configs.dvc_settings.dvc:
        dvc_data_path = configs.dvc_settings.dvc_parameters.dvc_data_path
        dvc_saver = DvcDataSaver()
        dvcs = dvc_saver.save_data(dvc_data_path)
        return dvcs
    else:
        pass


def neptune_logging(configs: Dict[str, Any], run):
    """
    Log training artifacts and parameters to Neptune.

    Args:
        configs (Dict[str, Any]): Configuration dictionary containing all settings.
        run: Neptune run object for logging parameters and artifacts.
    """
    try:
        run["params"] = configs
        if configs['dvc_settings']['dvc']:
            dvc_files = glob.glob(os.path.join(os.getcwd(), "*.dvc"))
            for dvc_file in dvc_files:
                print(dvc_file, "I am dvc file")
                run["data/" + os.path.basename(dvc_file)].upload(os.path.basename(dvc_file))
        model_path = os.path.join(configs['model']['model_save_folder'], run["sys/id"].fetch() + ".pkl")
        run["model/" + run["sys/id"].fetch() + ".pkl"].upload(model_path)
        log_filename = os.path.join(configs['logger']['folder'], run["sys/id"].fetch() + ".log")
        run['logs/' + os.path.basename(log_filename)].upload(log_filename)
        time.sleep(40)
        remove_files(run["sys/id"].fetch(), configs)
        return True
    except Exception as e:
        logging.error(f"Error during Neptune logging: {e}")
        raise


def s3_upload_artifact(artifact_path: str):
    """
    Upload model artifacts to S3.

    Args:
        artifact_path (str): Path to the artifact to be uploaded.
    """
    try:
        s3_handler = S3handler()
        s3_handler.s3suffix = EnvironmentVariables.MODEL_REGISTRY_REMOTE_SUFFIX.value
        s3_handler.upload_folder_or_file_to_s3(artifact_path)
        return True
    except Exception as e:
        logging.error(f"Error uploading artifact to S3: {e}")
        raise


def remove_all_except_latest(directory, latest_filename):
    """
    Removes all files in the specified directory except the latest file.

    Parameters:
    - directory (str): The directory from which to remove files.
    - latest_filename (str): The filename of the file to keep.

    Returns:
    None
    """
    if not os.path.isdir(directory):
        logging.error(f"Directory does not exist: {directory}")
        return

    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename != latest_filename:
                os.remove(file_path)
    except Exception as e:
        logging.error(f"Error removing files: {e}")


def remove_files(run_id, configs):
    """
    Removes specific files based on configurations and run ID.

    Parameters:
    - run_id (str): The current run ID to determine which files to keep.
    - configs (dict): Configuration dict containing paths and flags.

    Returns:
    None
    """
    log_dir = configs.get('logger', {}).get('folder', '')
    model_dir = configs.get('model', {}).get('model_save_folder', '')
    data_management = configs.get('data_management', {})
    path_to_processed = os.path.join(data_management.get('save_folder_processed', ''), 'processed.csv')
    path_to_original = os.path.join(data_management.get('save_folder_processed', ''), 'original.csv')

    if configs.get('clear_cache', {}).get('if_data', False):
        for path in [path_to_processed, path_to_original]:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logging.error(f"Error removing file {path}: {e}")

    if configs.get('clear_cache', {}).get('if_old_model_and_log', False):
        remove_all_except_latest(log_dir, f"{run_id}.log")
        remove_all_except_latest(model_dir, f"{run_id}.pkl")

    if configs.get('clear_cache', {}).get('if_dvc', False):

        if os.path.isfile('data.dvc'):
            try:
                os.remove('data.dvc')
                logging.info("data.dvc file is removed")
            except Exception as e:
                logging.error(f"Error removing file {'data.dvc'}: {e}")
    return True
