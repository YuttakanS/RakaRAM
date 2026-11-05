import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set modern styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Tahoma']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data(filepath: str = "data/processed/ram_dataset_real.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please run pipeline first.")
    df = pd.read_csv(filepath)
    df['price_per_gb'] = df['price_thb'] / df['capacity_gb']
    return df

def generate_eda_report(df: pd.DataFrame):
    print("=" * 60)
    print("        RakaRAM - EXPLORATORY DATA ANALYSIS (EDA)       ")
    print("=" * 60)
    
    # 1. Overview
    print(f"\n1. Overall Dataset Overview:")
    print(f"   - Total Samples: {len(df)}")
    print(f"   - DDR Generations: {dict(df['generation'].value_counts())}")
    print(f"   - Unique Brands ({df['brand'].nunique()}): {list(df['brand'].unique())}")
    print(f"   - Capacity Options: {sorted(df['capacity_gb'].unique())} GB")
    print(f"   - Speed Range: {df['speed_mhz'].min()} - {df['speed_mhz'].max()} MHz")
    print(f"   - Price Range: {df['price_thb'].min():,} - {df['price_thb'].max():,} THB")
    print(f"   - Median Price: {df['price_thb'].median():,.0f} THB")

    # 2. Generation Analysis
    print(f"\n2. DDR4 vs DDR5 Comparison:")
    gen_summary = df.groupby('generation').agg(
        count=('price_thb', 'count'),
        mean_price=('price_thb', 'mean'),
        median_price=('price_thb', 'median'),
        mean_price_per_gb=('price_per_gb', 'mean'),
        median_price_per_gb=('price_per_gb', 'median')
    )
    print(gen_summary.to_string())

    # 3. Brand Value Analysis
    print(f"\n3. Price per GB by Top Brands (Min 5 products):")
    top_brands = df['brand'].value_counts()[lambda x: x >= 5].index
    brand_summary = df[df['brand'].isin(top_brands)].groupby(['brand', 'generation']).agg(
        avg_price_per_gb=('price_per_gb', 'mean'),
        count=('price_thb', 'count')
    ).round(1)
    print(brand_summary.to_string())

    # 4. RGB Premium
    print(f"\n4. RGB Light Premium:")
    rgb_summary = df.groupby(['generation', 'has_rgb']).agg(
        mean_price=('price_thb', 'mean'),
        median_price=('price_thb', 'median'),
        count=('price_thb', 'count')
    )
    print(rgb_summary.to_string())

def plot_charts(df: pd.DataFrame):
    # Filter out extreme outliers for visualization clarity if needed
    q99 = df['price_thb'].quantile(0.98)
    df_plot = df[df['price_thb'] <= q99].copy()

    # Chart 1: Price Distribution by DDR Generation
    fig, ax = plt.subplots(figsize=(8, 5))
    ddr4_prices = df_plot[df_plot['generation'] == 'DDR4']['price_thb']
    ddr5_prices = df_plot[df_plot['generation'] == 'DDR5']['price_thb']
    ax.boxplot([ddr4_prices, ddr5_prices], tick_labels=['DDR4', 'DDR5'], patch_artist=True,
               boxprops=dict(facecolor='#3b82f6', color='#1d4ed8', alpha=0.7),
               medianprops=dict(color='#ef4444', linewidth=2))
    ax.set_title("RAM Price Distribution: DDR4 vs DDR5 (THB)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Price (THB)", fontsize=11)
    ax.yaxis.set_major_formatter('{x:,.0f}')
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/eda_01_ddr_price_distribution.png", dpi=150)
    plt.close()

    # Chart 2: Price per GB by Brand
    fig, ax = plt.subplots(figsize=(10, 5))
    top_brands = df['brand'].value_counts().head(8).index
    brand_df = df[df['brand'].isin(top_brands)].groupby('brand')['price_per_gb'].median().sort_values()
    colors = ['#10b981' if b in ['Lexar', 'KLEVV', 'TeamGroup'] else '#6366f1' for b in brand_df.index]
    bars = ax.barh(brand_df.index, brand_df.values, color=colors, alpha=0.85)
    ax.set_title("Median Price per GB by Brand (THB/GB)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("THB / GB", fontsize=11)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 5, bar.get_y() + bar.get_height()/2, f"{w:.1f} ฿/GB", va='center', fontsize=10)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/eda_02_price_per_gb_by_brand.png", dpi=150)
    plt.close()

    # Chart 3: Speed vs Price
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gen, color in [('DDR4', '#2563eb'), ('DDR5', '#ec4899')]:
        sub = df_plot[df_plot['generation'] == gen]
        ax.scatter(sub['speed_mhz'], sub['price_thb'], alpha=0.6, label=gen, color=color, s=40)
    ax.set_title("Speed (MHz) vs Price (THB)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Speed (MHz)", fontsize=11)
    ax.set_ylabel("Price (THB)", fontsize=11)
    ax.yaxis.set_major_formatter('{x:,.0f}')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/eda_03_speed_vs_price.png", dpi=150)
    plt.close()

    # Chart 4: RGB vs Non-RGB Price Difference
    fig, ax = plt.subplots(figsize=(8, 5))
    rgb_data = df_plot.groupby(['generation', 'has_rgb'])['price_thb'].median().unstack()
    rgb_data.columns = ['Non-RGB', 'RGB']
    rgb_data.plot(kind='bar', ax=ax, color=['#94a3b8', '#f59e0b'], alpha=0.9, rot=0)
    ax.set_title("Median Price Comparison: Non-RGB vs RGB (THB)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Median Price (THB)", fontsize=11)
    ax.yaxis.set_major_formatter('{x:,.0f}')
    ax.legend(["Non-RGB", "RGB (Lighting)"])
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/eda_04_rgb_price_impact.png", dpi=150)
    plt.close()

    # Chart 5: Correlation Heatmap
    num_cols = ['capacity_gb', 'modules', 'speed_mhz', 'cas_latency', 'has_rgb', 'price_thb']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='left')
    ax.set_yticklabels(num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.6 else "white", fontsize=9)
    ax.set_title("Correlation Matrix of RAM Features", fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/eda_05_correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nSuccessfully generated 5 EDA charts in '{FIGURES_DIR}/'")

if __name__ == "__main__":
    df = load_data()
    generate_eda_report(df)
    plot_charts(df)
