from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body, Query
from fastapi import APIRouter
from typing import Optional
import pickle
from src.helpers.prediction_pipeline import prediction_data_pipeline
import os
from src.utils.s3_handler import S3handler
from botocore.exceptions import NoCredentialsError, ClientError
import tempfile
import uuid
from urllib.parse import urlparse
import logging

router = APIRouter()


@router.get("/")
async def read_root():
    return {"message": "Welcome to the model evaluation API!"}


@router.post("/deploy-model/")
async def deploy_model(s3_model_uri: str = Query(..., description="s3_uri of the model to deploy")):
    # Validate the S3 URI format
    if not s3_model_uri.startswith("s3://"):
        logging.error("Invalid S3 URI format.")
        raise HTTPException(status_code=400, detail="Invalid S3 URI format.")
    try:
        s3_handler = S3handler()
        model_bucket, model_key = s3_handler.parse_s3_uri(s3_model_uri)

        # Clear any previous models in the directory
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logging.info(f"'{models_dir}' directory created")

        for f in os.listdir(models_dir):
            if f.endswith('.pkl'):
                os.remove(os.path.join(models_dir, f))
                logging.info(f"Removed old model: {f}")

        # Define the local path to save the new model
        local_file_path = os.path.join(models_dir, os.path.basename(model_key))
        # Download the new model
        s3_handler.download_file_from_s3(model_bucket, model_key, models_dir)
        logging.info(f"New model downloaded and saved to {local_file_path}")

        # Log the current model in the directory
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        logging.info(f"Available model files: {model_files}")

        return {"message": f"Model deployed successfully. Available model: {os.path.basename(local_file_path)}"}
    except NoCredentialsError:
        logging.error("AWS credentials not found.")
        raise HTTPException(status_code=500, detail="AWS credentials not found.")
    except ClientError as e:
        error_msg = f"AWS S3 Client Error: {e}"
        logging.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        logging.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/model/predict_file/")
async def predict_file(file: UploadFile = File(..., description="CSV input file"),
                       s3_model_uri: Optional[str] = Form(None, description="s3_uri of the model if the predictions "
                                                                            "are needed from a specific model"),
                       s3_output_uri: Optional[str] = Form(None, description="s3_uri if the output needs to be saved "
                                                                             "in S3"),
                       skip_feature_generator: bool = Form(False, description="Flag to skip feature "
                                                                              "extractor if already "
                                                                              "processed")):
    try:
        s3_handler = S3handler()
        if s3_model_uri:

            model_bucket, model_key = s3_handler.parse_s3_uri(s3_model_uri)
            temp_model_folder = "temp_models"
            os.makedirs('temp_models', exist_ok=True)
            s3_handler.download_file_from_s3(model_bucket, model_key, temp_model_folder)
            logging.info(f"Model downloaded from {s3_model_uri} to {temp_model_folder}")
            model_file_path = os.path.join(temp_model_folder, os.path.basename(model_key))
        else:
            if not os.path.exists('models'):
                raise HTTPException(status_code=400, detail="The system cannot find the path specified: 'models', "
                                                            "try deploying the model first")
            model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]

            if not model_files:
                raise HTTPException(status_code=400, detail="There is no model deployed, First deploy the model.")

            if len(model_files) != 1:
                raise HTTPException(status_code=400,
                                    detail="There should be exactly one .pkl file in the models folder.")

            model_file_path = os.path.join('models', model_files[0])

        with open(model_file_path, 'rb') as model_file:
            model = pickle.load(model_file)

        X, y = prediction_data_pipeline(file.file, skip_feature_generator)
        score = model.score(X, y)
        y_pred = model.predict(X)
        combined_df = X.copy()
        combined_df['Y_test'] = y
        combined_df['Y_pred'] = y_pred
        combined_json = combined_df.to_dict(orient="records")
        if s3_output_uri:
            csv_file_name = f"predictions_{uuid.uuid4()}.csv"
            temp_csv_path = os.path.join(tempfile.gettempdir(), csv_file_name)
            combined_df.to_csv(temp_csv_path, index=False)
            parsed_uri = urlparse(s3_output_uri)
            s3_handler.s3bucketname = parsed_uri.netloc
            s3_handler.s3suffix = parsed_uri.path.lstrip('/')

            s3_handler.upload_folder_or_file_to_s3(temp_csv_path)
            logging.info(f"Combined DataFrame uploaded to S3 bucket '{s3_output_uri}' as {csv_file_name}")
            os.remove(temp_csv_path)

        if s3_model_uri:
            os.remove(model_file_path)
            logging.info(f"Temporary model file {model_file_path} removed.")

        return {"model_score": score, "combined_data": combined_json}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
