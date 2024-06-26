from src.utils.config_parser import create_config
from src.helpers.training_pipeline import (
    training_data_pipeline,
    get_model,
    dvc_handler,
    neptune_logging,
    s3_upload_artifact,
)
from src.logger.loggers import get_logger
import argparse
import os
import logging
import neptune.integrations.sklearn as npt_utils
import matplotlib.pyplot as plt
import pickle


def setup_logging(configs, run_id):
    """Set up logging to file and console."""
    log_filename = os.path.join(configs["logger"]["folder"], f"{run_id}.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    return log_filename


def main(args):
    configs = create_config(args.config_path)
    plt.switch_backend("agg")
    run = get_logger(configs)
    run_id = run["sys/id"].fetch()

    log_filename = setup_logging(configs, run["sys/id"].fetch())
    logging.info("Training pipeline started.")

    X_train, X_test, y_train, y_test = training_data_pipeline(
        configs, args.skip_feature_generator
    )

    model = get_model(configs)
    model.fit(X_train, y_train)

    if configs.get("model", {}).get("parameter_search_enabled", False):
        model = model.best_estimator_

    run["rfr_summary"] = npt_utils.create_regressor_summary(
        model, X_train, X_test, y_train, y_test
    )

    logging.info("Saving model...")
    model_dir = configs.get("model", {}).get("model_save_folder", ".")
    model_save_path = os.path.join(model_dir, f"{run_id}.pkl")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    pickle.dump(model, open(model_save_path, "wb"))
    logging.info(f"Model is saved in {model_save_path} file")
    s3_upload_artifact(model_save_path)
    dvc_handler(configs)
    neptune_logging(configs, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the training pipeline with a configuration file."
    )

    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="src/configs/config.yaml",
        help="Path to the configuration file in JSON format.",
    )
    parser.add_argument(
        "-s",
        "--skip_feature_generator",
        action="store_true",
        help="Skip running the feature generator.",
    )

    args = parser.parse_args()
    main(args)
