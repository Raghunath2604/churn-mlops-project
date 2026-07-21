import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

def train_model(X, y, config: dict):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_state"]
    )
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=config["random_state"],
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Test accuracy: {acc:.4f}")
    return model, acc
