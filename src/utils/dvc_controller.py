import os
import configparser
import glob
import logging
from typing import Union, List
from dvc.repo import Repo
from src.configs.enums import EnvironmentVariables


class DvcDataSaver:
    def __init__(self, remote_name: str = "myremote") -> None:
        self.remote_name = remote_name
        self.remote_url = EnvironmentVariables.TRAINING_UNIVERSE_DVC_CACHE.value
        self.dvc_repo = self.initialize_dvc_repo()

    def initialize_dvc_repo(self) -> Repo:
        """Initialize DVC repo if not already initialized and configure DVC remote."""
        try:
            if not os.path.exists(".dvc"):
                repo = Repo.init(no_scm=True)
                logging.info("DVC repository initialized.")
            else:
                repo = Repo()
            # Configure DVC remote if not already configured
            self.configure_dvc_remote(repo)
            return repo
        except Exception as e:
            logging.error(f"Error initializing DVC repository: {e}")
            raise

    def configure_dvc_remote(self, repo: Repo):
        """Configure DVC remote storage."""
        config_path = os.path.join(repo.root_dir, '.dvc', 'config')
        config = configparser.ConfigParser()

        try:
            config.read(config_path)
            remote_section = f'remote "{self.remote_name}"'

            if not config.has_section(remote_section):
                config.add_section(remote_section)
            config.set(remote_section, 'url', self.remote_url)

            if 'core' not in config.sections():
                config.add_section('core')
            config.set('core', 'remote', self.remote_name)

            with open(config_path, 'w') as configfile:
                config.write(configfile)
                logging.info(f"Remote '{self.remote_name}' configured as default in DVC.")
        except Exception as e:
            logging.error(f"Error configuring DVC remote: {e}")
            raise

    def save_data(self, source: Union[str, List[str]]) -> List[str]:
        """Save data to DVC and push it to the remote storage."""
        try:
            sources = [source] if isinstance(source, str) else source
            dvc_files_created = []

            for src in sources:
                dvc_file = f"{src}.dvc"
                self.dvc_repo.add(src)
                self.dvc_repo.push(dvc_file, remote=self.remote_name)
                dvc_files_created.append(dvc_file)

            logging.info("Data saved and pushed to DVC remote.")
            return dvc_files_created
        except Exception as e:
            logging.error(f"Error saving data to DVC: {e}")
            raise


# Example usage setup corrected for demonstration purposes:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    remote_name = "myremote"  # Example remote name

    data_saver = DvcDataSaver(remote_name)
    # Adjust the source path according to your project structure
    source_path = "path/to/your/data"
    data_saver.save_data(source_path)
