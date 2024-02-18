from typing import Union, List
import os
from dvc.repo import Repo
from src.configs.enums import EnviromentVariables
import configparser

class BaseDataSaver:
    def __init__(self, save_location: str) -> None:
        self.save_location = save_location
        # Additional initialization as needed


class DvcDataSaver(BaseDataSaver):
    def __init__(self, save_location: str, remote_name: str = "myremote") -> None:
        super().__init__(save_location)
        self.remote_name = remote_name
        self.remote_url = EnviromentVariables.TRAINING_UNIVERSE_DVC_CACHE.value
        self.dvc_repo = self.initialize_dvc_repo()

    def initialize_dvc_repo(self):
        """Initialize DVC repo if not already initialized."""
        if not os.path.exists(".dvc"):
            self.repo = Repo.init(no_scm=True)
            print("DVC repository initialized.")
        else:
            self.repo = Repo()

        # Configure DVC remote if not already configured and remote_url provided
        if self.remote_url:
            self.configure_dvc_remote()

        return self.repo

    def configure_dvc_remote(self):
        """Configure DVC remote storage by directly modifying the .dvc/config file."""
        config_path = os.path.join(self.repo.root_dir, '.dvc', 'config')
        config = configparser.ConfigParser()

        # Ensure the config file exists
        if not os.path.exists(config_path):
            print("DVC config file does not exist. Are you sure the repository is initialized?")
            return

        config.read(config_path)

        # Check if the remote is already configured
        remote_section = f'remote "{self.remote_name}"'
        if config.has_section(remote_section) and config.get(remote_section, 'url', fallback=None) == self.remote_url:
            print(f"Remote '{self.remote_name}' already exists with the configured URL.")
        else:
            # Add or update the remote section
            if not config.has_section(remote_section):
                config.add_section(remote_section)
            config.set(remote_section, 'url', self.remote_url)

            # Optionally, set this remote as the default
            if 'core' not in config.sections():
                config.add_section('core')
            config.set('core', 'remote', self.remote_name)

            # Write the updated configuration back to the file
            with open(config_path, 'w') as configfile:
                config.write(configfile)
                print(f"Remote '{self.remote_name}' added or updated and set as default.")

    def save_data(self, source: Union[str, List[str]]):
        """Save data directory in dvc file and push it to dvc storage location."""
        if isinstance(source, str):
            self.dvc_repo.add(source)
            self.dvc_repo.push(f"{source}.dvc", remote=self.remote_name)
        else:
            for src in source:
                self.dvc_repo.add(src)
                self.dvc_repo.push(f"{src}.dvc", remote=self.remote_name)
        print("Data saved and pushed to DVC remote.")
        return


# Example usage:
if __name__ == "__main__":
    # Configure these variables as needed
    save_location = r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook\training_universe"

    data_saver = DvcDataSaver(save_location)
    data_saver.save_data(r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook\training_universe\requirements.txt")
