import argparse, logging, pickle, yaml
from churn_model.data import load_data, clean_data
from churn_model.features import select_features
from churn_model.train import train_model

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: {config}")
    df = load_data(config["data_path"])
    df = clean_data(df)
    X, y = select_features(df)
    model, acc = train_model(X, y, config)
    with open(config["model_output_path"], "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {config['model_output_path']}")

if __name__ == "__main__":
    main()
