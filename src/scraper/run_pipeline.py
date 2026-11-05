import json
import os
import pandas as pd
from datetime import datetime
from src.scraper.jib_scraper import JibRamScraper
from src.parser.spec_extractor import parse_ram_product
from src.utils.logger import get_logger

logger = get_logger("run_pipeline")

def run_data_pipeline(save_raw: bool = True, output_csv: str = "data/processed/ram_dataset_real.csv") -> pd.DataFrame:
    """
    Executes the full data collection & feature extraction pipeline.
    """
    logger.info(">>> Starting RakaRAM Data Ingestion Pipeline <<<")
    
    # Ensure target directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 1. Scrape raw listings
    scraper = JibRamScraper(request_delay=0.4)
    raw_products = scraper.scrape_all()

    if not raw_products:
        logger.error("No products were scraped! Pipeline aborted.")
        return pd.DataFrame()

    # 2. Save raw snapshot
    if save_raw:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_filepath = f"data/raw/ram_raw_snapshot_{timestamp}.json"
        with open(raw_filepath, "w", encoding="utf-8") as f:
            json.dump(raw_products, f, ensure_ascii=False, indent=2)
        # Also maintain a latest pointer
        with open("data/raw/ram_raw_latest.json", "w", encoding="utf-8") as f:
            json.dump(raw_products, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(raw_products)} raw items to {raw_filepath}")

    # 3. Parse and extract specs
    logger.info("Parsing specifications from product titles...")
    parsed_items = [parse_ram_product(item) for item in raw_products]
    df = pd.DataFrame(parsed_items)

    # 4. Data Cleaning & Validation
    initial_count = len(df)
    # Drop rows without capacity or unknown generation
    df = df.dropna(subset=['capacity_gb', 'price_thb'])
    df['capacity_gb'] = df['capacity_gb'].astype(int)
    df = df[df['generation'].isin(['DDR4', 'DDR5'])]
    df = df[df['price_thb'] > 300]  # Filter out anomalies/accessories
    df = df.drop_duplicates(subset=['name', 'price_thb'])
    
    cleaned_count = len(df)
    logger.info(f"Cleaning complete: {initial_count} raw -> {cleaned_count} valid RAM records.")

    # 5. Save processed CSV
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Saved processed dataset to {output_csv}")

    # 6. Summary Report
    print("\n" + "=" * 50)
    print("       RakaRAM DATASET SUMMARY REPORT       ")
    print("=" * 50)
    print(f"Total Products : {len(df)}")
    print(f"Generations    : {dict(df['generation'].value_counts())}")
    print(f"Top Brands     : {dict(df['brand'].value_counts().head(6))}")
    print(f"Price Range    : {df['price_thb'].min():,} - {df['price_thb'].max():,} THB (Mean: {df['price_thb'].mean():,.0f} THB)")
    print("=" * 50 + "\n")

    return df

if __name__ == "__main__":
    run_data_pipeline()
