"""
End-to-end pipeline runner:
- Download/load data
- Preprocess and persist train/test splits and preprocessor
- Train & select best model with threshold tuning
- Evaluate and write metrics/plots
"""
import joblib
import yaml
from pathlib import Path

from src.data_prep import load_from_kagglehub, prepare_data, build_preprocessor, split_data
from src.utils import ensure_dir
from src.train import train_and_select
from src.evaluate import evaluate


def preprocess_and_save(cfg):
    df = load_from_kagglehub(cfg["dataset"]["kagglehub_id"])
    X, y, cat_cols, num_cols = prepare_data(df, cfg["dataset"]["target_col"])
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train, X_test, y_train, y_test = split_data(
        X, y, cfg["training"]["test_size"], cfg["training"]["random_state"]
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    joblib.dump(preprocessor, processed_dir / "preprocessor.joblib")
    joblib.dump(X_train_t, processed_dir / "X_train.joblib")
    joblib.dump(X_test_t, processed_dir / "X_test.joblib")
    y_train.to_frame("left").to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_frame("left").to_csv(processed_dir / "y_test.csv", index=False)

    print("Saved processed data to", processed_dir)


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    preprocess_and_save(cfg)
    train_and_select(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
