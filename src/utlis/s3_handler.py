import boto3
import os
from src.configs.enums import EnviromentVariables


class S3handler:

    def __init__(self):

        self.s3bucketname = EnviromentVariables.BUCKET_NAME.value
        self.s3suffix = EnviromentVariables.TRAINING_UNIVERSE_REMOTE_SUFFIX.value
        self.s3 = boto3.client("s3")

    def download_one_file_from_s3(self, bucket, prefix_and_key, local_path=os.getcwd()):
        """
        this function downloads one file from s3 to local
        :param bucket: (str) bucket to download from
        :param prefix_and_key: (str) prefix of file to download
        :param local_path: (str) path to download file in
        :return: (str) path of downloaded file
        """
        self.s3.download_file(bucket, prefix_and_key, os.path.join(local_path, prefix_and_key.split('/')[-2]))
        return os.path.join(local_path, prefix_and_key.split('/')[-2])

    @staticmethod
    def split_s3_path(s3_path):
        """
        this function splits s3 URI to bucket, prefix, and file name
        :param s3_path: (str) s3 URI format s3://bucket/prefix/file.extension
        :return: (tuple) contains bucket, prefix, and file name
        """
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        prefix = "/".join(path_parts[0:])
        return bucket, prefix

    def upload_folder_or_file_to_s3(self, inputDir: str):
        """
        uploads directory or file to s3 directory

        :param inputDir: the file or folder to be uploaded
        :type inputDir: str
        :raises e: failed to upload
        """

        print("Uploading results to s3 initiated...")


        try:
            if os.path.isfile(inputDir):
                print("upload : ", inputDir, " to Target: ", self.s3suffix, end="")
                self.s3.upload_file(inputDir, self.s3bucketname, self.s3suffix.replace("\\", "/"))
                print(" ...Success")
            else:
                for path, _, files in os.walk(inputDir):
                    for file in files:
                        dest_path = path.replace(inputDir, "")
                        __s3file = os.path.normpath(self.s3suffix + "/" + dest_path + "/" + file)
                        __local_file = os.path.join(path, file)
                        print("upload : ", __local_file, " to Target: ", __s3file, end="")
                        self.s3.upload_file(
                            __local_file, self.s3bucketname, __s3file.replace("\\", "/")
                        )
                        print(" ...Success")
        except Exception as e:
            print(" ... Failed!! Quitting Upload!!")
            print(e)
            raise e


if __name__ == "__main__":
    s3_handler = S3handler()
    input_dir = r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook\training_universe"

    model_s3_uri = "s3://project-main/training-universe/model/requirements.txt"
    s3_handler.s3suffix = os.path.join(s3_handler.s3suffix, 'model')
    s3_handler.upload_folder_or_file_to_s3(input_dir)
    local_path = r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook"
    model_bucket, model_prefix = s3_handler.split_s3_path(model_s3_uri)
    print(model_bucket, model_prefix,"model_bucket, model_prefix")
    model_checkpoint_path = s3_handler.download_one_file_from_s3(model_bucket, model_prefix, local_path)
