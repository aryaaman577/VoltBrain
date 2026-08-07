import os
import sys
import time
import math
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, explained_variance_score, mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(__file__))
from preprocess import prepare_data

def train_and_evaluate():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'EV_Specs_Dataset.csv')
    models_dir = os.path.join(BASE_DIR, 'models')
    charts_dir = os.path.join(BASE_DIR, 'assets', 'charts')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("Dataset not found.")
        sys.exit(1)

    raw_df = pd.read_csv(data_path)
    rows_count = len(raw_df)
    cols_count = len(raw_df.columns)

    X, y, preprocessor = prepare_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Extra Trees Regressor': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    param_grids = {
        'Random Forest Regressor': {
            'model__n_estimators': [100, 150, 200],
            'model__max_depth': [12, 18, 25, None],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        },
        'Extra Trees Regressor': {
            'model__n_estimators': [100, 150, 200],
            'model__max_depth': [12, 18, 25, None],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        },
        'Gradient Boosting Regressor': {
            'model__n_estimators': [100, 150, 200],
            'model__learning_rate': [0.03, 0.05, 0.1, 0.15],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 0.9, 1.0]
        }
    }

    comparison_results = []
    candidates = []

    for name, model in base_models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        t_train = time.time() - t0

        t0 = time.time()
        preds = pipeline.predict(X_test)
        t_pred = time.time() - t0

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

        comparison_results.append({
            'Model': name,
            'Version': 'Default',
            'Tuned (Yes/No)': 'No',
            'MAE': round(float(mae), 4),
            'RMSE': round(float(rmse), 4),
            'R2': round(float(r2), 4),
            'TrainingTime': round(float(t_train), 4),
            'PredictionTime': round(float(t_pred), 4)
        })

        candidates.append({
            'name': f"{name} (Default)",
            'pipeline': pipeline,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            't_train': t_train,
            't_pred': t_pred,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'best_params': "None (Default)",
            'preds': preds
        })

        if name in param_grids:
            search = RandomizedSearchCV(
                estimator=Pipeline([('preprocessor', preprocessor), ('model', base_models[name])]),
                param_distributions=param_grids[name],
                n_iter=10,
                cv=5,
                scoring='r2',
                random_state=42,
                n_jobs=-1
            )

            t0 = time.time()
            search.fit(X_train, y_train)
            t_train_tuned = time.time() - t0

            tuned_pipeline = search.best_estimator_

            t0 = time.time()
            preds_tuned = tuned_pipeline.predict(X_test)
            t_pred_tuned = time.time() - t0

            mae_t = mean_absolute_error(y_test, preds_tuned)
            rmse_t = np.sqrt(mean_squared_error(y_test, preds_tuned))
            r2_t = r2_score(y_test, preds_tuned)

            cv_scores_t = cross_val_score(tuned_pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

            clean_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}

            comparison_results.append({
                'Model': name,
                'Version': 'Tuned',
                'Tuned (Yes/No)': 'Yes',
                'MAE': round(float(mae_t), 4),
                'RMSE': round(float(rmse_t), 4),
                'R2': round(float(r2_t), 4),
                'TrainingTime': round(float(t_train_tuned), 4),
                'PredictionTime': round(float(t_pred_tuned), 4)
            })

            candidates.append({
                'name': f"{name} (Tuned)",
                'pipeline': tuned_pipeline,
                'mae': mae_t,
                'rmse': rmse_t,
                'r2': r2_t,
                't_train': t_train_tuned,
                't_pred': t_pred_tuned,
                'cv_mean': np.mean(cv_scores_t),
                'cv_std': np.std(cv_scores_t),
                'best_params': clean_params,
                'preds': preds_tuned
            })

    best_cand = None
    best_r2 = -float('inf')
    best_mae = float('inf')

    for cand in candidates:
        r2_val = cand['r2']
        mae_val = cand['mae']
        if r2_val > best_r2 + 1e-4:
            best_r2 = r2_val
            best_mae = mae_val
            best_cand = cand
        elif math.isclose(r2_val, best_r2, abs_tol=1e-4) and mae_val < best_mae:
            best_r2 = r2_val
            best_mae = mae_val
            best_cand = cand

    comparison_df = pd.DataFrame(comparison_results)
    comparison_csv_path = os.path.join(models_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)

    model_save_path = os.path.join(models_dir, 'voltbrain_pipeline.pkl')
    joblib.dump(best_cand['pipeline'], model_save_path)

    reloaded_pipeline = joblib.load(model_save_path)
    test_preds_orig = best_cand['pipeline'].predict(X_test)
    test_preds_reloaded = reloaded_pipeline.predict(X_test)
    if not np.allclose(test_preds_orig, test_preds_reloaded):
        print("Serialization error: Reloaded pipeline predictions differ!")
        sys.exit(1)

    edge_cases = pd.DataFrame({
        'Battery Capacity (kWh)': [20.0, 150.0, 85.0, 75.0, 60.0, 90.0],
        'Efficiency (Wh/km)': [110.0, 280.0, 200.0, 280.0, 160.0, 210.0],
        'Top Speed (km/h)': [120.0, 280.0, 180.0, 200.0, 150.0, 220.0],
        'Brand': ['Tesla', 'Porsche', 'CustomBrandX', 'Tesla', 'UnknownBrandY', 'Audi'],
        'Drive Type': ['FWD', 'AWD', 'RWD', 'TriMotorSpecial', 'FWD', 'QuadDrive']
    })
    
    try:
        stress_preds = reloaded_pipeline.predict(edge_cases)
        if len(stress_preds) != len(edge_cases):
            print("Stress test failed: Prediction count mismatch.")
            sys.exit(1)
    except Exception as e:
        print(f"Stress test error: {e}")
        sys.exit(1)

    best_preds = best_cand['preds']
    med_ae = float(median_absolute_error(y_test, best_preds))
    exp_var = float(explained_variance_score(y_test, best_preds))
    mape_val = float(mean_absolute_percentage_error(y_test, best_preds))

    report = {
        "Dataset Rows": rows_count,
        "Dataset Columns": cols_count,
        "Best Model": best_cand['name'],
        "Best Parameters": best_cand['best_params'],
        "Cross Validation Mean": round(float(best_cand['cv_mean']), 4),
        "Cross Validation Std": round(float(best_cand['cv_std']), 4),
        "Training Time": round(float(best_cand['t_train']), 4),
        "Prediction Time": round(float(best_cand['t_pred']), 4),
        "MAE": round(float(best_cand['mae']), 4),
        "RMSE": round(float(best_cand['rmse']), 4),
        "R2": round(float(best_cand['r2']), 4),
        "Median Absolute Error": round(med_ae, 4),
        "Explained Variance Score": round(exp_var, 4),
        "MAPE": round(mape_val, 4),
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Scikit-Learn Version": sklearn.__version__,
        "Training Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    report_path = os.path.join(models_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    plt.style.use('default')

    fitted_model = best_cand['pipeline'].named_steps['model']
    fitted_preprocessor = best_cand['pipeline'].named_steps['preprocessor']

    if hasattr(fitted_model, 'feature_importances_'):
        feature_names = fitted_preprocessor.get_feature_names_out()
        importances = fitted_model.feature_importances_
        clean_names = [name.split('__')[-1] for name in feature_names]
        indices = np.argsort(importances)[-10:]

        plt.figure(figsize=(8, 5), facecolor='white')
        plt.barh([clean_names[i] for i in indices], importances[indices], color='#2563EB')
        plt.title('Top 10 Feature Importances', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Importance Score', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, 'feature_importance.png'), dpi=300, facecolor='white')
        plt.close()

    plt.figure(figsize=(7, 5), facecolor='white')
    plt.scatter(y_test, best_preds, alpha=0.6, color='#2563EB', edgecolors='none')
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#EF4444', linestyle='--', linewidth=1.5)
    plt.title('Predicted vs Actual Driving Range', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Actual Range (km)', fontsize=10)
    plt.ylabel('Predicted Range (km)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'prediction_vs_actual.png'), dpi=300, facecolor='white')
    plt.close()

    residuals = y_test - best_preds
    plt.figure(figsize=(7, 5), facecolor='white')
    plt.scatter(best_preds, residuals, alpha=0.6, color='#10B981', edgecolors='none')
    plt.axhline(0, color='#EF4444', linestyle='--', linewidth=1.5)
    plt.title('Residual Plot (Errors vs Predictions)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Predicted Range (km)', fontsize=10)
    plt.ylabel('Residual Error (km)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'residual_plot.png'), dpi=300, facecolor='white')
    plt.close()

    num_df = raw_df.select_dtypes(include=[np.number])
    corr_matrix = num_df.corr()
    plt.figure(figsize=(8, 6), facecolor='white')
    cax = plt.matshow(corr_matrix, cmap='coolwarm', fignum=1)
    plt.colorbar(cax)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='left', fontsize=8)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns, fontsize=8)
    plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold', pad=24)
    plt.savefig(os.path.join(charts_dir, 'correlation_heatmap.png'), dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

    train_sizes, train_scores, test_scores = learning_curve(
        best_cand['pipeline'], X, y, cv=5, scoring='r2', n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(7, 5), facecolor='white')
    plt.plot(train_sizes, train_mean, 'o-', color='#2563EB', label='Training Score')
    plt.plot(train_sizes, test_mean, 'o-', color='#10B981', label='Cross-Validation Score')
    plt.title('Learning Curve (R² vs Training Size)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Training Set Size', fontsize=10)
    plt.ylabel('R² Score', fontsize=10)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'learning_curve.png'), dpi=300, facecolor='white')
    plt.close()

    print("Robustness execution finished.")
    print(f"Verified Pipeline Model: {best_cand['name']}")

if __name__ == "__main__":
    train_and_evaluate()
