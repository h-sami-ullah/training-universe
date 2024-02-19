import boto3
import os
from src.configs.enums import EnvironmentVariables
import logging
from typing import Tuple


class S3handler:

    def __init__(self):
        """
         Initializes the S3Handler object with bucket name and suffix from environment variables,
         and creates a boto3 S3 client.
         """

        self.s3bucketname = EnvironmentVariables.BUCKET_NAME.value
        self.s3suffix = EnvironmentVariables.TRAINING_UNIVERSE_REMOTE_SUFFIX.value
        self.s3 = boto3.client("s3")

    def download_file_from_s3(self, bucket_name: str, object_key: str, local_download_path: str = None) -> str:
        """
        Downloads a single file from an S3 bucket to a local directory.

        Args:
            bucket_name: The name of the S3 bucket to download the file from.
            object_key: The S3 object key (including prefix) of the file to download.
            local_download_path: The local directory path where the file will be downloaded.
                                 If not provided, the current working directory will be used.

        Returns:
            The full local path of the downloaded file.
        """

        # Use current working directory if no download path is provided
        if local_download_path is None:
            local_download_path = os.getcwd()

        # Construct the full local file path
        local_file_path = os.path.join(local_download_path, os.path.basename(object_key))

        try:
            # Attempt to download the file
            logging.info(f"Downloading '{object_key}' from bucket '{bucket_name}' to '{local_file_path}'")
            self.s3.download_file(Bucket=bucket_name, Key=object_key, Filename=local_file_path)
            logging.info("Download successful.")
        except Exception as e:
            logging.error(f"Failed to download file: {e}")
            raise

        return local_file_path

    @staticmethod
    def split_s3_path(s3_path: str) -> Tuple[str, str]:
        """
        Splits an S3 URI into bucket, prefix, and file name.

        Args:
            s3_path: S3 URI in the format s3://bucket/prefix/file.extension.

        Returns:
            A tuple containing bucket, prefix, and file name.
        """
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts[0:])
        return bucket, prefix

    def upload_folder_or_file_to_s3(self, input_path: str) -> None:
        """
        Uploads a directory or file to an S3 bucket.

        Args:
            input_path: The file or folder to be uploaded.
        """
        logging.info("Uploading results to S3 initiated...")

        try:
            if os.path.isfile(input_path):
                self._upload_file_to_s3(input_path)
            else:
                self._upload_directory_to_s3(input_path)
        except Exception as e:
            logging.error("Failed to upload to S3. Quitting upload process.", exc_info=True)
            raise

    def _upload_file_to_s3(self, file_path: str) -> None:
        """
        Helper method to upload a single file to S3.

        Args:
            file_path: Path of the file to upload.
        """
        filename = os.path.basename(file_path)
        s3_dest_path = os.path.join(self.s3suffix, filename).replace("\\", "/")
        logging.info(f"Uploading {file_path} to Target: {s3_dest_path}")
        self.s3.upload_file(file_path, self.s3bucketname, s3_dest_path)
        logging.info("Upload successful.")

    def _upload_directory_to_s3(self, directory_path: str) -> None:
        """
        Helper method to upload a directory to S3.

        Args:
            directory_path: Path of the directory to upload.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, directory_path)
                s3_dest_path = os.path.join(self.s3suffix, relative_path).replace("\\", "/")

                logging.info(f"Uploading {local_file_path} to Target: {s3_dest_path}")
                self.s3.upload_file(local_file_path, self.s3bucketname, s3_dest_path)
                logging.info("Upload successful.")


if __name__ == "__main__":
    s3_handler = S3handler()
    input_dir = r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook\training_universe"

    model_s3_uri = "s3://project-main/training-universe/model/requirements.txt"
    s3_handler.s3suffix = os.path.join(s3_handler.s3suffix, 'model')
    s3_handler.upload_folder_or_file_to_s3(input_dir)
    local_path = r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook"
    model_bucket, model_prefix = s3_handler.split_s3_path(model_s3_uri)
    print(model_bucket, model_prefix, "model_bucket, model_prefix")
    model_checkpoint_path = s3_handler.download_one_file_from_s3(model_bucket, model_prefix, local_path)
