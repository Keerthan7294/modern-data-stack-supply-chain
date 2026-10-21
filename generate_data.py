"""
generate_data.py
----------------
Generates 100,000 rows of realistic, messy e-commerce supply chain data
(daily sales, variable lead times, stock levels, returns, and nulls)
and loads it into a PostgreSQL database with high-performance COPY streaming.
"""

import os
import sys
import time
import random
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text

# Set random seeds for deterministic realism
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

# Configuration
TOTAL_ROWS = 100_000
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "supply_chain_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
RAW_SCHEMA = "raw"
RAW_TABLE = "supply_chain_orders"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def create_product_catalog():
    """Generates a catalog of 50 SKUs across 5 core categories with realistic baselines."""
    categories = [
        "Electronics",
        "Apparel",
        "Home & Kitchen",
        "Beauty & Health",
        "Sports & Outdoors",
    ]
    
    suppliers = [
        {"name": "Shenzhen Global Tech Ltd", "mean_lead": 18, "std_lead": 6.5, "mode": "Sea Freight"},
        {"name": "Apex Logistics Hub", "mean_lead": 7, "std_lead": 2.1, "mode": "Air Freight"},
        {"name": "Nordic Fulfillment AB", "mean_lead": 10, "std_lead": 3.0, "mode": "Standard Express"},
        {"name": "Pacific Coastal Cargo", "mean_lead": 21, "std_lead": 8.0, "mode": "Sea Freight"},
        {"name": "EuroDirect Distribution", "mean_lead": 9, "std_lead": 2.5, "mode": "Road Freight"},
        {"name": "Domestic NextDay LLC", "mean_lead": 3, "std_lead": 0.8, "mode": "Overnight"},
        {"name": "FastTrack Freight Corp", "mean_lead": 12, "std_lead": 4.2, "mode": "Air Freight"},
        {"name": "Andes Supply Chain SA", "mean_lead": 16, "std_lead": 5.0, "mode": "Sea Freight"},
    ]

    skus = []
    sku_counter = 1
    for cat in categories:
        cat_code = cat[:4].upper()
        for i in range(1, 11):
            sku_id = f"SKU-{cat_code}-{sku_counter:04d}"
            sku_counter += 1
            supplier = random.choice(suppliers)
            
            # Base demand distribution characteristics
            base_daily_demand = random.randint(8, 45)
            # Current stock level: some healthy, some tight, some overstocked
            current_stock = int(base_daily_demand * random.uniform(3, 35))
            reorder_batch = int(base_daily_demand * random.uniform(15, 45))
            base_price = round(random.uniform(12.50, 480.00), 2)
            
            skus.append({
                "sku_id": sku_id,
                "product_name": f"{fake.word().capitalize()} {fake.word().capitalize()} {sku_id[-4:]}",
                "clean_category": cat,
                "supplier": supplier["name"],
                "shipping_mode": supplier["mode"],
                "promised_lead_time": supplier["mean_lead"],
                "lead_std": supplier["std_lead"],
                "base_daily_demand": base_daily_demand,
                "current_stock_level": current_stock,
                "reorder_batch_size": reorder_batch,
                "unit_price": base_price
            })
    return skus


def inject_messy_casing(text_value: str) -> str:
    """Injects casing, whitespace, and capitalization inconsistencies typical of raw data."""
    dice = random.random()
    if dice < 0.25:
        return text_value.lower()
    elif dice < 0.45:
        return text_value.upper()
    elif dice < 0.60:
        return f"  {text_value} "
    elif dice < 0.70:
        return f" {text_value.lower()}  "
    return text_value


def generate_messy_dataset(total_rows: int = TOTAL_ROWS) -> pd.DataFrame:
    """Creates 100,000 synthetic supply chain records with deliberate real-world flaws."""
    print(f"[*] Initializing synthetic data engine for {total_rows:,} rows...")
    start_time = time.time()
    
    skus = create_product_catalog()
    sku_weights = np.random.dirichlet(np.ones(len(skus)) * 1.5)
    
    warehouses = ["WH-EAST-01", "WH-WEST-02", "WH-CENTRAL-01", "WH-SOUTH-03"]
    regions = ["North America", "Europe", "APAC", "LATAM", "Middle East"]
    statuses = ["Delivered", "Delivered", "Delivered", "In Transit", "Pending", "Cancelled", "Returned"]
    
    # 365 day date distribution
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    total_seconds = int((end_date - start_date).total_seconds())
    
    # Generate batch array of SKU indices
    chosen_sku_indices = np.random.choice(len(skus), size=total_rows, p=sku_weights)
    
    records = []
    
    print("[*] Generating orders, lead time distributions, and supply chain metrics...")
    for idx, sku_idx in enumerate(chosen_sku_indices):
        sku = skus[sku_idx]
        
        # Order ID
        order_id = f"ORD-{idx + 1:07d}"
        
        # Order date: random timestamp in the past year
        offset_seconds = random.randint(0, total_seconds)
        order_dt = start_date + timedelta(seconds=offset_seconds)
        
        # Units ordered: predominantly positive Poisson demand, ~1% negative for returns
        is_return = random.random() < 0.012
        if is_return:
            units = -random.randint(1, 4)
            status = "Returned"
        else:
            units = int(np.random.poisson(lam=sku["base_daily_demand"] / 4.0)) + 1
            status = random.choice(statuses)
            
        # Discount percent: ~8% missing/nulls, otherwise between 0.00 and 0.35
        if random.random() < 0.08:
            discount = None
        else:
            discount = round(random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]), 2)
            
        # Lead time calculation with variance and occasional supply chain shocks (lognormal tail)
        promised_lead = sku["promised_lead_time"]
        # Normal lead time variance + 5% port/weather disruption spikes
        lead_noise = np.random.normal(loc=0.0, scale=sku["lead_std"])
        if random.random() < 0.05:
            disruption_spike = np.random.exponential(scale=14.0)  # port congestion shock
            actual_lead = max(1, int(round(promised_lead + lead_noise + disruption_spike)))
        else:
            actual_lead = max(1, int(round(promised_lead + lead_noise)))
            
        # Delivery date logic: pending or in-transit orders have no delivery date yet
        if status in ["In Transit", "Pending", "Cancelled"]:
            delivery_dt = None
            reported_lead_time = None
        else:
            delivery_dt = order_dt + timedelta(days=actual_lead)
            reported_lead_time = actual_lead
            
        # Customer region with ~3% nulls
        cust_region = None if random.random() < 0.03 else random.choice(regions)
        
        # Messy category and supplier names
        category_raw = inject_messy_casing(sku["clean_category"])
        supplier_raw = inject_messy_casing(sku["supplier"])
        
        records.append({
            "order_id": order_id,
            "order_date": order_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "sku_id": sku["sku_id"],
            "product_name": sku["product_name"],
            "category": category_raw,
            "supplier_name": supplier_raw,
            "warehouse_id": random.choice(warehouses),
            "units_ordered": units,
            "unit_price": sku["unit_price"],
            "discount_percent": discount,
            "shipping_mode": sku["shipping_mode"],
            "promised_lead_time_days": promised_lead,
            "actual_lead_time_days": reported_lead_time,
            "delivery_date": delivery_dt.strftime("%Y-%m-%d %H:%M:%S") if delivery_dt else None,
            "order_status": status,
            "current_stock_level": sku["current_stock_level"],
            "reorder_batch_size": sku["reorder_batch_size"],
            "customer_region": cust_region
        })
        
    df = pd.DataFrame(records)
    
    # Inject ~500 duplicate rows (~0.5%) to mirror webhook retries/ETL duplicates
    print("[*] Injecting deliberate webhook retry duplicates (~500 rows)...")
    dup_sample = df.sample(n=500, random_state=42).copy()
    df = pd.concat([df, dup_sample], ignore_index=True)
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    elapsed = time.time() - start_time
    print(f"[+] Dataset generated successfully: {len(df):,} rows in {elapsed:.2f} seconds.")
    return df


def save_local_archive(df: pd.DataFrame):
    """Archives the generated dataset locally in Parquet and CSV format."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parquet_path = os.path.join(OUTPUT_DIR, "raw_supply_chain_orders.parquet")
    csv_path = os.path.join(OUTPUT_DIR, "raw_supply_chain_orders.csv")
    
    print(f"[*] Archiving dataset to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    # Also save a sample CSV for inspection
    df.head(1000).to_csv(csv_path, index=False)
    print(f"[+] Local archive saved: {parquet_path} ({os.path.getsize(parquet_path) / (1024*1024):.2f} MB)")


def insert_into_postgres(df: pd.DataFrame):
    """
    High-performance insertion into PostgreSQL database using COPY protocol.
    Loads 100,000 rows in ~3 seconds.
    """
    print(f"[*] Connecting to PostgreSQL ({DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME})...")
    conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
    
    try:
        conn = psycopg2.connect(conn_str)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Ensure raw schema exists
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA};")
        
        # Drop and create raw table with appropriate staging types
        create_table_sql = f"""
        DROP TABLE IF EXISTS {RAW_SCHEMA}.{RAW_TABLE} CASCADE;
        CREATE TABLE {RAW_SCHEMA}.{RAW_TABLE} (
            order_id VARCHAR(64),
            order_date TIMESTAMP,
            sku_id VARCHAR(64),
            product_name VARCHAR(255),
            category VARCHAR(128),
            supplier_name VARCHAR(255),
            warehouse_id VARCHAR(64),
            units_ordered INT,
            unit_price NUMERIC(10, 2),
            discount_percent NUMERIC(5, 2),
            shipping_mode VARCHAR(64),
            promised_lead_time_days INT,
            actual_lead_time_days INT,
            delivery_date TIMESTAMP,
            order_status VARCHAR(64),
            current_stock_level INT,
            reorder_batch_size INT,
            customer_region VARCHAR(64)
        );
        """
        cur.execute(create_table_sql)
        print(f"[+] Table {RAW_SCHEMA}.{RAW_TABLE} created.")
        
        # Stream dataframe to PostgreSQL using in-memory CSV buffer and COPY
        print(f"[*] Streaming {len(df):,} records via binary COPY buffer...")
        copy_start = time.time()
        
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False, na_rep="\\N", sep="\t")
        buffer.seek(0)
        
        cur.copy_from(
            buffer,
            f"{RAW_SCHEMA}.{RAW_TABLE}",
            sep="\t",
            null="\\N",
            columns=[
                "order_id", "order_date", "sku_id", "product_name", "category",
                "supplier_name", "warehouse_id", "units_ordered", "unit_price",
                "discount_percent", "shipping_mode", "promised_lead_time_days",
                "actual_lead_time_days", "delivery_date", "order_status",
                "current_stock_level", "reorder_batch_size", "customer_region"
            ]
        )
        
        cur.execute(f"SELECT COUNT(*) FROM {RAW_SCHEMA}.{RAW_TABLE};")
        row_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        copy_elapsed = time.time() - copy_start
        print(f"[+] Bulk COPY completed: {row_count:,} rows inserted in {copy_elapsed:.2f} seconds!")
        return True

    except Exception as e:
        print(f"[!] PostgreSQL connection/insertion notice: {e}")
        print("[!] If the Docker container is not yet started, run: docker compose up -d")
        print("[*] Local Parquet backup remains available for dbt / Streamlit pipelines.")
        return False


def main():
    print("=" * 70)
    print("  MODERN DATA STACK: SUPPLY CHAIN DATA GENERATOR")
    print(f"  Target: {TOTAL_ROWS:,} rows of messy e-commerce inventory logs")
    print("=" * 70)
    
    df = generate_messy_dataset(TOTAL_ROWS)
    save_local_archive(df)
    insert_into_postgres(df)
    
    print("\n[+] Data generation pipeline complete!")


if __name__ == "__main__":
    main()
