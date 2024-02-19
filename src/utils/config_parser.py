import os
import yaml
from easydict import EasyDict as edict


def create_config(exp_file):

    with open(exp_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Copy all the arguments
    cfg = edict()
    for k, v in config.items():
        cfg[k] = v

    return cfg



if __name__=="__main__":
    path_to_config = r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook\training_universe\src\configs\training_config.yaml"
    create_config(path_to_config)