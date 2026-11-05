"""
RakaRAM - Linear Regression Analysis Module
===========================================
Performs comprehensive Linear Regression modeling on real RAM market data:
1. Simple Linear Regression (Price vs Capacity by DDR Generation)
2. Multiple Linear Regression (OLS) with Coefficients, StdErr, t-stat, p-values
3. Log-Transformed Linear Regression & Ridge Comparison
4. 5-Fold Cross-Validation & Test Set Evaluation
5. Publication-quality Diagnostic Visualizations (Actual vs Predicted, Residuals, Coefficients Bar Chart)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT_DIR / "data" / "processed" / "ram_dataset_real.csv"

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run pipeline first.")
    df = pd.read_csv(DATA_PATH)
    return df

def run_simple_linear_regression(df: pd.DataFrame):
    print("=" * 70)
    print("   1. SIMPLE LINEAR REGRESSION (Price vs. Capacity)   ")
    print("=" * 70)

    # Overall
    X_cap = df[['capacity_gb']]
    y = df['price_thb']
    model_all = LinearRegression().fit(X_cap, y)
    r2_all = r2_score(y, model_all.predict(X_cap))
    print(f"Overall Model:")
    print(f"  Equation : Price (THB) = {model_all.intercept_:.2f} + {model_all.coef_[0]:.2f} * Capacity(GB)")
    print(f"  R² Score : {r2_all:.4f}")
    print(f"  Slope interpretation : เพิ่มความจุ 1 GB ราคาเพิ่มขึ้นเฉลี่ย {model_all.coef_[0]:.2f} บาท\n")

    # DDR4 vs DDR5 breakdown
    gen_models = {}
    for gen in ['DDR4', 'DDR5']:
        sub = df[df['generation'] == gen]
        if len(sub) == 0:
            continue
        m = LinearRegression().fit(sub[['capacity_gb']], sub['price_thb'])
        r2 = r2_score(sub['price_thb'], m.predict(sub[['capacity_gb']]))
        gen_models[gen] = (m, r2)
        print(f"{gen} Sub-model ({len(sub)} samples):")
        print(f"  Equation : Price (THB) = {m.intercept_:.2f} + {m.coef_[0]:.2f} * Capacity(GB)")
        print(f"  R² Score : {r2:.4f}")
        print(f"  Rate per GB : ~{m.coef_[0]:.2f} THB/GB\n")

    return model_all, gen_models

def run_multiple_linear_regression(df: pd.DataFrame):
    print("=" * 70)
    print("   2. MULTIPLE LINEAR REGRESSION (OLS with Full Statistical Inference)   ")
    print("=" * 70)

    cat_cols = ['brand', 'generation', 'form_factor']
    num_cols = ['capacity_gb', 'modules', 'speed_mhz', 'cas_latency', 'has_rgb']
    
    ct = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

    X_raw = df[cat_cols + num_cols]
    y = df['price_thb'].values
    X_mat = ct.fit_transform(X_raw)

    cat_feature_names = ct.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    feature_names = cat_feature_names + num_cols

    # Add intercept column
    X_design = np.column_stack([np.ones(len(X_mat)), X_mat])
    col_names = ['Intercept'] + feature_names

    n, p = X_design.shape

    # Solve OLS: beta = (X'X)^(-1) X'y
    inv_xtx = np.linalg.pinv(X_design.T @ X_design)
    beta = inv_xtx @ X_design.T @ y
    y_pred = X_design @ beta
    residuals = y - y_pred

    # Degrees of freedom and variance
    df_resid = n - p
    s2 = np.sum(residuals ** 2) / df_resid
    std_err = np.sqrt(np.diag(s2 * inv_xtx))
    t_stat = beta / std_err
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_resid))

    # Confidence intervals (95%)
    t_crit = stats.t.ppf(0.975, df=df_resid)
    ci_lower = beta - t_crit * std_err
    ci_upper = beta + t_crit * std_err

    summary_df = pd.DataFrame({
        'Feature': col_names,
        'Coefficient': beta,
        'Std Error': std_err,
        't-Statistic': t_stat,
        'p-Value': p_values,
        '[95% Conf. Interval]': [f"[{l:,.1f}, {u:,.1f}]" for l, u in zip(ci_lower, ci_upper)]
    })

    print(summary_df.to_string(index=False))

    r2 = r2_score(y, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (df_resid)
    mae = mean_absolute_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)

    print("\nModel Fit Metrics (Full Sample):")
    print(f"  Observations (N)   : {n}")
    print(f"  Parameters (k)     : {p}")
    print(f"  R² Score           : {r2:.4f}")
    print(f"  Adjusted R²        : {adj_r2:.4f}")
    print(f"  MAE                : {mae:,.2f} THB")
    print(f"  RMSE               : {rmse:,.2f} THB")

    return summary_df, beta, col_names, y, y_pred, residuals, ct

def evaluate_cross_validation_models(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("   3. MODEL COMPARISON: 5-FOLD CROSS VALIDATION   ")
    print("=" * 70)

    cat_cols = ['brand', 'generation', 'form_factor']
    num_cols = ['capacity_gb', 'modules', 'speed_mhz', 'cas_latency', 'has_rgb']
    X = df[cat_cols + num_cols]
    y = df['price_thb']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])

    models = {
        "Multiple Linear Regression (Raw)": LinearRegression(),
        "Multiple Linear Regression (Log)": TransformedTargetRegressor(
            regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1
        ),
        "Ridge Regression (alpha=1.5, Raw)": Ridge(alpha=1.5),
        "Ridge Regression (alpha=1.5, Log)": TransformedTargetRegressor(
            regressor=Ridge(alpha=1.5), func=np.log1p, inverse_func=np.expm1
        )
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print(f"{'Model Architecture':<36} | {'CV R²':<10} | {'CV MAE (THB)':<14} | {'CV RMSE (THB)':<14}")
    print("-" * 80)

    for name, model in models.items():
        pipe = Pipeline([
            ('prep', preprocessor),
            ('reg', model)
        ])
        cv_res = cross_validate(
            pipe, X, y, cv=cv,
            scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
        )
        r2_m = cv_res['test_r2'].mean()
        mae_m = -cv_res['test_neg_mean_absolute_error'].mean()
        rmse_m = -cv_res['test_neg_root_mean_squared_error'].mean()
        results.append({
            'Model': name,
            'R2': r2_m,
            'MAE': mae_m,
            'RMSE': rmse_m
        })
        print(f"{name:<36} | {r2_m:<10.4f} | {mae_m:<14,.1f} | {rmse_m:<14,.1f}")

    print("=" * 80)
    return pd.DataFrame(results)

def plot_regression_diagnostics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray, summary_df: pd.DataFrame):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RakaRAM - Linear Regression Analysis & Diagnostics', fontsize=16, fontweight='bold', y=0.98)

    # 1. Actual vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.65, edgecolors='none', color='#0B5D46', s=50)
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    min_val = min(y_true.min(), y_pred.min()) * 0.95
    ax1.plot([min_val, max_val], [min_val, max_val], color='#D9534F', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')
    ax1.set_title('1. Actual vs. Predicted Price (THB)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Actual Price (THB)', fontsize=11)
    ax1.set_ylabel('Predicted Price (THB)', fontsize=11)
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    r2_val = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.90, f'R² = {r2_val:.4f}\nMAE = {mean_absolute_error(y_true, y_pred):,.0f} THB', 
             transform=ax1.transAxes, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#ccc'))
    ax1.legend(loc='lower right')

    # 2. Residual Plot
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuals, alpha=0.65, color='#286090', s=50)
    ax2.axhline(0, color='#D9534F', linestyle='--', linewidth=1.5)
    ax2.set_title('2. Residuals vs. Fitted Values', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Price (THB)', fontsize=11)
    ax2.set_ylabel('Residual (Actual - Pred) THB', fontsize=11)

    # 3. Simple Regression: Capacity vs Price by Generation
    ax3 = axes[1, 0]
    colors = {'DDR4': '#2E7D32', 'DDR5': '#1565C0'}
    for gen in ['DDR4', 'DDR5']:
        sub = df[df['generation'] == gen]
        if len(sub) > 0:
            ax3.scatter(sub['capacity_gb'], sub['price_thb'], label=f'{gen} Data', color=colors[gen], alpha=0.6, s=45)
            # fit line
            line_m = LinearRegression().fit(sub[['capacity_gb']], sub['price_thb'])
            x_line = pd.DataFrame({'capacity_gb': np.linspace(sub['capacity_gb'].min(), sub['capacity_gb'].max(), 50)})
            y_line = line_m.predict(x_line)
            ax3.plot(x_line['capacity_gb'], y_line, color=colors[gen], linewidth=2.5, 
                     label=f'{gen} Trend ({line_m.coef_[0]:.1f} THB/GB)')
    ax3.set_title('3. Simple Regression: Capacity vs. Price by Generation', fontsize=12, fontweight='bold')
    ax3.set_xlabel('RAM Capacity (GB)', fontsize=11)
    ax3.set_ylabel('Price (THB)', fontsize=11)
    ax3.legend(loc='upper left')

    # 4. Coefficients Bar Chart
    ax4 = axes[1, 1]
    # Filter out Intercept for clean relative comparison
    coef_plot_df = summary_df[summary_df['Feature'] != 'Intercept'].copy()
    coef_plot_df['Feature_Clean'] = coef_plot_df['Feature'].str.replace('brand_', 'Brand: ').str.replace('generation_', 'Gen: ').str.replace('form_factor_', 'Form: ')
    coef_plot_df = coef_plot_df.sort_values(by='Coefficient', ascending=True)

    bar_colors = ['#2E7D32' if c > 0 else '#C62828' for c in coef_plot_df['Coefficient']]
    ax4.barh(coef_plot_df['Feature_Clean'], coef_plot_df['Coefficient'], color=bar_colors, alpha=0.85)
    ax4.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax4.set_title('4. Multiple Linear Regression Coefficients (Impact in THB)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Coefficient Value (THB)', fontsize=11)

    plt.tight_layout()
    plot_path = FIGURES_DIR / "linear_regression_analysis.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved diagnostic visualization plots to: {plot_path}")

def main():
    df = load_dataset()
    print(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns.\n")

    run_simple_linear_regression(df)
    summary_df, beta, col_names, y, y_pred, residuals, ct = run_multiple_linear_regression(df)
    evaluate_cross_validation_models(df)
    plot_regression_diagnostics(df, y, y_pred, residuals, summary_df)

if __name__ == "__main__":
    main()
