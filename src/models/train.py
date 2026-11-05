import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.logger import get_logger

logger = get_logger("model_training")

CATEGORICAL_FEATURES = ['brand', 'generation', 'form_factor']
NUMERICAL_FEATURES = ['capacity_gb', 'modules', 'speed_mhz', 'cas_latency', 'has_rgb']
TARGET = 'price_thb'

def build_preprocessor() -> ColumnTransformer:
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    num_transformer = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, CATEGORICAL_FEATURES),
            ('num', num_transformer, NUMERICAL_FEATURES)
        ]
    )

def evaluate_models(X: pd.DataFrame, y: pd.Series):
    preprocessor = build_preprocessor()
    
    # We use TransformedTargetRegressor with log1p/expm1 to prevent negative predictions
    # and model percentage-based price variations naturally
    base_models = {
        "Random Forest": RandomForestRegressor(n_estimators=180, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.07, max_depth=4, random_state=42),
        "Ridge Regression": Ridge(alpha=1.5)
    }

    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    logger.info("Evaluating candidate models with 5-Fold Cross Validation (Log Target)...")
    print("\n" + "=" * 65)
    print("      MODEL CROSS-VALIDATION COMPARISON (5-Fold CV)      ")
    print("=" * 65)
    print(f"{'Model':<20} | {'R2 Score':<12} | {'MAE (THB)':<12} | {'RMSE (THB)':<12}")
    print("-" * 65)

    best_name = None
    best_score = -float('inf')
    best_pipeline = None

    for name, base_reg in base_models.items():
        transformed_reg = TransformedTargetRegressor(
            regressor=base_reg,
            func=np.log1p,
            inverse_func=np.expm1
        )
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', transformed_reg)
        ])
        
        cv_res = cross_validate(
            pipeline, X, y, cv=cv,
            scoring={
                'r2': 'r2',
                'neg_mae': 'neg_mean_absolute_error',
                'neg_rmse': 'neg_root_mean_squared_error'
            }
        )

        r2_mean = float(cv_res['test_r2'].mean())
        mae_mean = float(-cv_res['test_neg_mae'].mean())
        rmse_mean = float(-cv_res['test_neg_rmse'].mean())

        results[name] = {
            'r2_mean': r2_mean,
            'mae_mean': mae_mean,
            'rmse_mean': rmse_mean
        }

        print(f"{name:<20} | {r2_mean:<12.4f} | {mae_mean:<12.1f} | {rmse_mean:<12.1f}")

        # Choose model with best R2 (and lowest MAE)
        if r2_mean > best_score:
            best_score = r2_mean
            best_name = name
            best_pipeline = pipeline

    print("=" * 65 + "\n")
    logger.info(f"Selected model: {best_name} (R2: {best_score:.4f})")
    return best_name, base_models[best_name], results

def train_and_save(data_path: str = "data/processed/ram_dataset_real.csv", output_dir: str = "saved_models"):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading processed dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Prepare features and target
    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = df[features].copy()
    y = df[TARGET].copy()

    # Train/Test split for final test validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    logger.info(f"Split data: {len(X_train)} train samples, {len(X_test)} test samples.")

    # Cross-validation & selection
    best_name, best_regressor, cv_metrics = evaluate_models(X_train, y_train)

    # Train final pipeline on training set
    preprocessor = build_preprocessor()
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', TransformedTargetRegressor(
            regressor=best_regressor,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])
    final_pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = final_pipeline.predict(X_test)
    test_r2 = float(r2_score(y_test, y_pred))
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    logger.info(f"Held-out Test Evaluation - R2: {test_r2:.4f}, MAE: {test_mae:.1f} THB, RMSE: {test_rmse:.1f} THB")

    # Fit on all data for production deployment
    final_pipeline.fit(X, y)

    model_path = os.path.join(output_dir, "ram_price_model.joblib")
    joblib.dump(final_pipeline, model_path)
    logger.info(f"Saved production model pipeline to {model_path}")

    # Save metadata
    metadata = {
        "model_name": best_name,
        "saved_at": datetime.now().isoformat(),
        "total_training_samples": len(df),
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES
        },
        "supported_brands": sorted(df['brand'].unique().tolist()),
        "supported_generations": sorted(df['generation'].unique().tolist()),
        "metrics": {
            "cv_5fold": cv_metrics,
            "test_set": {
                "r2": test_r2,
                "mae": test_mae,
                "rmse": test_rmse
            }
        }
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}")

    return final_pipeline, metadata

if __name__ == "__main__":
    train_and_save()
