import argparse, logging, sys
import mlflow, yaml
from mlflow import MlflowClient
from churn_model.data import load_data, clean_data
from churn_model.features import select_features
from churn_model.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
MODEL_NAME = "churn-classifier"
ALIAS = "champion"

def get_champion_accuracy(client):
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        run = mlflow.get_run(champion.run_id)
        return run.data.metrics.get("accuracy"), champion.version
    except Exception:
        return None, None

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-prediction")
    client = MlflowClient()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    champion_acc, champion_version = get_champion_accuracy(client)
    if champion_acc is not None:
        logger.info(f"Current champion: version {champion_version}, accuracy {champion_acc:.4f}")
    else:
        logger.info("No champion registered yet")

    with mlflow.start_run():
        mlflow.log_params({"n_estimators": config["n_estimators"], "max_depth": config["max_depth"]})
        df = load_data(config["data_path"]); df = clean_data(df)
        X, y = select_features(df)
        model, acc = train_model(X, y, config)
        from sklearn.model_selection import train_test_split
        _, _, _, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])
        baseline_acc = y_test.value_counts(normalize=True).max()
        mlflow.log_metric("baseline_accuracy", baseline_acc)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

    logger.info(f"Candidate run {run_id}: accuracy {acc:.4f} (baseline: {baseline_acc:.4f})")
    if acc <= baseline_acc:
        logger.warning(f"REJECTED: candidate ({acc:.4f}) does not beat trivial baseline ({baseline_acc:.4f})")
        return 0
    if champion_acc is None or acc > champion_acc:
        result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, ALIAS, result.version)
        logger.info(f"PROMOTED: version {result.version} (accuracy {acc:.4f}) is the new champion")
    else:
        logger.info(f"REJECTED: candidate ({acc:.4f}) did not beat champion ({champion_acc:.4f})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
