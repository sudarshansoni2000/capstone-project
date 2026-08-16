import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")



# -------------------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "sudarshansoni2000"
repo_name = "capstone-project"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise
def register_model(model_name: str, model_info: dict):
    try:
        model_uri = model_info["model_uri"]

        logging.info(
            "Registering model from: %s",
            model_uri
        )

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logging.info(
            "Model %s version %s registered successfully",
            model_name,
            model_version.version
        )

    except Exception as e:
        logging.exception(
            "Error during model registration: %s",
            e
        )
        raise

# def register_model(model_name: str, model_info: dict):
#     """Register model in MLflow Model Registry."""

#     try:

#         run_id = model_info["run_id"]
#         model_path = model_info["model_path"]

#         model_uri = f"runs:/{run_id}/{model_path}"

#         logging.info("Registering model from: %s", model_uri)

#         client = mlflow.tracking.MlflowClient()

#         artifacts = client.list_artifacts(run_id)

#         logging.info(
#             "Artifacts found: %s",
#             [artifact.path for artifact in artifacts]
#         )

#         model_version = mlflow.register_model(
#             model_uri=model_uri,
#             name=model_name
#         )

#         logging.info(
#             "Model %s version %s registered successfully",
#             model_name,
#             model_version.version
#         )

#     except Exception as e:

#         logging.exception(
#             "Error during model registration: %s",
#             e
#         )

#         raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
