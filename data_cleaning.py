import os
import re
from datetime import datetime
import pandas as pd


def clean_ecommerce_data(file_path):
    """
    Loads and cleans the ecommerce dataset:
    - removes administrative/non-product stock codes (including gift_/POST/BANK/AMAZONFEE),
    - filters descriptions with administrative keywords,
    - separates cancellations (InvoiceNo starting with 'C'),
    - enforces Quantity > 0 and UnitPrice > 0 for sales,
    - parses InvoiceDate, drops rows with invalid dates,
    - drops rows missing CustomerID, deduplicates,
    - logs if Saturday is missing from the sales dates (investigate separately).
    """
    def _read_input_file(path):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            # fallback CSV (same basename .csv)
            csv_path = os.path.splitext(path)[0] + '.csv'
            if os.path.exists(csv_path):
                print("Parquet engine unavailable — reading CSV fallback:", csv_path)
                return pd.read_csv(csv_path)
            raise RuntimeError(
                "Failed to read parquet. Install pyarrow or fastparquet, e.g.:\n"
                "  python -m pip install pyarrow\n"
                "or provide a CSV at: " + csv_path
            ) from e

    df = _read_input_file(file_path)
    print(f"Initial shape: {df.shape}")

    # Normalize text columns to strings
    df['StockCode'] = df['StockCode'].astype(str).str.strip()
    df['Description'] = df['Description'].astype(str).str.strip()

    # 1. Remove Non-Product Stock Codes (exact and prefix-based)
    non_product_exact = {
        'POST', 'D', 'M', 'BANK CHARGE', 'AMAZONFEE', 'CRUK', 'DOT', 'S',
        'gift_0001_20'
    }
    # prefixes that indicate fees/labels rather than SKUs
    non_product_prefixes = ('gift_', 'post', 'bank', 'amazonfee', 'fee', 'cruk')

    mask_exact = df['StockCode'].isin(non_product_exact)
    mask_prefix = df['StockCode'].str.lower().str.startswith(non_product_prefixes)
    df = df[~(mask_exact | mask_prefix)]

    # 2. Remove Administrative Descriptions (expanded keywords + word boundaries)
    error_keywords = [
        r'\bmanual\b', r'\badjust', r'\berror\b', r'\btest\b',
        r'wrongly coded', r'incorrectly credited', r'\bbroken\b',
        r'\bfound\b', r'thrown away', r'gift card', r'postage', r'bank charge'
    ]
    desc_pattern = re.compile('|'.join(error_keywords), flags=re.IGNORECASE)
    df = df[~df['Description'].astype(str).str.contains(desc_pattern, na=False)]

    # 3. Separate Cancellations (InvoiceNo starting with 'C')
    cancellations = df[df['InvoiceNo'].astype(str).str.startswith('C', na=False)].copy()
    df_sales = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)].copy()

    # 4. Ensure numeric fields and filter invalid values
    df_sales['Quantity'] = pd.to_numeric(df_sales['Quantity'], errors='coerce')
    if 'UnitPrice' in df_sales.columns:
        df_sales['UnitPrice'] = pd.to_numeric(df_sales['UnitPrice'], errors='coerce')

    df_sales = df_sales[df_sales['Quantity'] > 0]
    if 'UnitPrice' in df_sales.columns:
        df_sales = df_sales[df_sales['UnitPrice'] > 0]

    # 5. Drop missing Customer IDs (for customer-centric analysis)
    if 'CustomerID' in df_sales.columns:
        df_sales = df_sales.dropna(subset=['CustomerID'])

    # 6. Parse InvoiceDate and drop rows with invalid dates
    if 'InvoiceDate' in df_sales.columns:
        df_sales['InvoiceDate'] = pd.to_datetime(df_sales['InvoiceDate'], errors='coerce')
        invalid_dates = df_sales['InvoiceDate'].isna().sum()
        if invalid_dates:
            print(f"Dropping {invalid_dates} rows with invalid InvoiceDate")
            df_sales = df_sales.dropna(subset=['InvoiceDate'])

    # 7. Deduplicate
    before_dup = len(df_sales)
    df_sales = df_sales.drop_duplicates()
    print(f"Dropped {before_dup - len(df_sales)} duplicate rows")

    # 8. Check for missing Saturday (log for investigation)
    if 'InvoiceDate' in df_sales.columns:
        days_present = set(df_sales['InvoiceDate'].dt.day_name().unique())
        if 'Saturday' not in days_present:
            print("Warning: Saturday is missing from sales dates — investigate export/operational schedule")

    print(f"Cleaned Sales shape: {df_sales.shape}")
    print(f"Cancellations shape: {cancellations.shape}")

    return df_sales, cancellations


def save_cleaned_data(sales_df, cancellations_df, out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)

    sales_parquet = os.path.join(out_dir, 'cleaned_ecommerce.parquet')
    try:
        sales_df.to_parquet(sales_parquet, index=False)
        print(f"Saved cleaned data: {sales_parquet}")
    except Exception as e:
        print(f"Parquet save skipped or failed: {e}")


if __name__ == "__main__":
    input_path = os.path.join('data', 'ecommerce.parquet')
    sales_df, cancellations_df = clean_ecommerce_data(input_path)
    save_cleaned_data(sales_df, cancellations_df, out_dir='data')