from fastapi import FastAPI
import logging
import os
import uvicorn
from api.app import create_app

app: FastAPI = create_app()


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


def main():
    setup_logging()
    logging.info("API Starting")
    PORT = 5005
    if 'PORT' in os.environ.keys():
        PORT = int(os.environ['PORT'])

    uvicorn.run(app, host="0.0.0.0", port=PORT, loop="auto", log_level='info')
    logging.info("API Stopping")


if __name__ == "__main__":
    main()
