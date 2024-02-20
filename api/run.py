from fastapi import FastAPI
import logging
from api.routes.prediction import router as prediction_router
import os
import uvicorn


def create_api_desc()->str:
    desc=''
    for line in open(r'README.md','r').readlines():
        desc+=line+'\n'
    return desc

def setup_logging():
    """Set up logging to file and console."""
    log_filename = os.path.join("api_log", "logs.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    # Configure root logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename),
                                  logging.StreamHandler()])
    return log_filename

app:FastAPI = FastAPI(title="Restaurant Hotspot Prediction Service",
                      description=create_api_desc())
def main():
    setup_logging()

    app.include_router(prediction_router)
    uvicorn.run(app, host="0.0.0.0", port=5005)



if __name__ == "__main__":
    main()


