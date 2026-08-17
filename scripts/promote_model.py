# promote_model.py

import os
import mlflow
import dagshub


def promote_model():
    # ---------------------------------------------------------
    # 1. Set up DagsHub credentials
    # ---------------------------------------------------------

    dagshub_token = os.getenv("CAPSTONE_TEST")

    if not dagshub_token:
        raise EnvironmentError(
            "CAPSTONE_TEST environment variable is not set"
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "sudarshansoni2000"
    repo_name = "capstone-project"

    # ---------------------------------------------------------
    # 2. Set MLflow tracking URI
    # ---------------------------------------------------------

    # mlflow.set_tracking_uri(
    #     "https://dagshub.com/sudarshansoni2000/capstone-project.mlflow"
    # )
    # dagshub.init(repo_owner='sudarshansoni2000', repo_name='capstone-project', mlflow=True)

    # mlflow.set_tracking_uri(
    #     f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    # )

    # ---------------------------------------------------------
    # 3. Create MLflow client
    # ---------------------------------------------------------

    client = mlflow.MlflowClient()

    model_name = "my_model"

    # ---------------------------------------------------------
    # 4. Get the model currently marked as @candidate
    # ---------------------------------------------------------

    try:
        candidate_model = client.get_model_version_by_alias(
            name=model_name,
            alias="candidate"
        )

    except Exception as e:
        raise RuntimeError(
            f"No @candidate alias found for model '{model_name}'. "
            "Make sure the registered model is assigned the "
            "@candidate alias before promotion."
        ) from e

    candidate_version = candidate_model.version

    print(
        f"Candidate model found: "
        f"{model_name} version {candidate_version}"
    )

    # ---------------------------------------------------------
    # 5. Promote candidate by assigning @champion alias
    # ---------------------------------------------------------

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=candidate_version
    )

    # ---------------------------------------------------------
    # 6. Confirmation
    # ---------------------------------------------------------

    print(
        f"Model {model_name} version {candidate_version} "
        f"promoted to @champion"
    )


if __name__ == "__main__":
    promote_model()