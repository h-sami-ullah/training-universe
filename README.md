# Geo-Busyness Predictor 
## (Training Universe)



## Table of Contents
- [Project Overview](#1-project-overview)
- [Objectives](#2-objectives)
- [Implementation Details](#3-implementation-details)
- [Getting Started](#4-getting-started)
- [Usage](#5-usage)
  - [Training](#51--training)
  - [Inference](#52--inference)
    - [Using Docker](#521--using-docker)
    - [Without Docker](#522--without-docker)
- [Contributing](#6-contributing)
- [License](#7-license)


## 1. Project Overview
This project aims at developing a machine learning model that 
predicts the busyness of geographical regions based on courier locations 
during food collection at restaurants. Utilizing geolocation data and h3 
hexagons for regional definition, this model provides insights into the 
activity levels of different areas, assisting in optimizing logistics and 
operations for delivery services.


## 2. Objectives
- **Refactor the PoC**: Transform the POC into a structured, maintainable, and scalable ML pipeline.
- **Enhance Code Quality**: Employ object-oriented programming and software engineering best practices to improve code modularity and readability.
- **Operational Efficiency**: Implement a continuous integration and deployment (CI/CD) workflow to streamline updates and deployment.

## 3. Implementation Details
The project is structured around four main components:
- `data_collection.py`: Basic handler for data loading
- `feature_generation.py`: Feature Extractor modules.
- `training.py`: Training Entry Point
- `prediction.py`: Manages the prediction process using the trained model.

Additionally, the project incorporates:
- A configuration system for easy management of model parameters and settings.
- Dependency management to ensure consistent environments across development and production.
- It uses below tech stack
  - `Docker` for containerization
  - `AWS S3` for data storage
  - `Bitbucket` for CI/CD
  - `Local` runner for pipline execution 
  - `DVC` for data versioning
  - `Neptune` for Experiment Tracking
  - `ECS` for API hosting


## 4. Getting Started
To set up the project locally:
```
conda create --name py38 python=3.8     # set up a virtual environment with python 3.8 
git clone <repository-url>              # Clone the repository
cd local/path/to/repository             
pip install -r requirements.txt         # Installs all required depencies

```


## 5. Usage
### 5.1- Training
The following files need to be adapted in order to run the code on your own machine:
- Change the `file_path` in `data_collection_settings` in  `src/configs/config.yaml` file to your csv file
- Change the rest of the parameters as per your need, config the logger, and set flags if you want to remove cache
- In `src/configs/enums.py` set your EnvironmentVariables needed for s3
- `src/training.py` is main entry point for training it can take two arguments 
  - a config file if you have a new config file
  - a `--skip_feature_generator` flag to skip feature extraction incase if the file is already processed
- Once all set you can start the training with `python -m src.training` with optional arguments if needed
- Once the training finishes the model will be logged model artifact will be logged in `S3` as well as `neptune`, in S3 it will be in `MODEL_REGISTRY_REMOTE_SUFFIX` set in `src/configs/enums.py` and in neptune under model experiment number

### 5.2- Inference

The inference is hosted as an API, to run API

#### 5.2.1- Using Docker
* pull the repo using `git clone <repository-url>`
* `make PORT=5005 image_tag=training-universe`
* to only build the training-universe image, run `make build PORT=5005 image_tag=training-universe`
* to only deploy the training-universe service, run `make deploy PORT=5005 image_tag=training-universe`
#### 5.2.2- Without Docker
* Set up the environment as per described in 


## 6. Contributing
Contributions to improve the model's accuracy, efficiency, or code quality are welcome. Please refer to the contributing guidelines for more details.

## 7. License
Specify your project's license here, ensuring compliance with the data usage policies and legal requirements.


