from fastapi import FastAPI
from api.routes.prediction import router as prediction_router


def create_api_desc() -> str:
    desc = ''
    for line in open(r'README.md', 'r').readlines():
        desc += line + '\n'
    return desc


def create_app() -> FastAPI:
    app: FastAPI = FastAPI(title="Restaurant Hotspot Prediction Service",
                           description=create_api_desc())

    app.include_router(prediction_router)

    return app
