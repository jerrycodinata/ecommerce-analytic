import sys
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, to_timestamp, month, sum, avg, count, countDistinct,
    datediff, max, lit, lag, hour, dayofweek, when, abs, ntile
)

# --- CONFIGURATION ---
HDFS_INPUT_PATH = "hdfs://namenode:9000/data/ecommerce/cleaned_ecommerce.parquet"
OUTPUT_DIR_BASE = "/opt/scripts/results"

# --- HELPER FUNCTIONS ---
def save_output(df, name):
    """Helper function to save a DataFrame as a single Parquet file."""
    output_path = f"{OUTPUT_DIR_BASE}/{name}"
    print(f"--- Saving '{name}' to {output_path}...")
    df.repartition(1).write.mode("overwrite").parquet(output_path)
    print(f"--- Successfully saved '{name}'.")


# --- ANALYSIS 1: RFM (Recency, Frequency, Monetary) ---
def analyze_rfm(df):
    """
    Calculates RFM scores for each customer.
    - Recency: Days since last purchase.
    - Frequency: Total number of distinct transactions.
    - Monetary: Total sales value.
    Scores (1-5) are calculated using ntile (5-quintiles).
    """
    print("Starting RFM Analysis...")
    
    # 1. Find the most recent date in the entire dataset to use as a "snapshot" date.
    snapshot_date = df.select(max(col("InvoiceTimestamp"))).first()[0]
    
    # 2. Calculate R, F, and M
    rfm_df = df.groupBy("CustomerID").agg(
        datediff(lit(snapshot_date), max("InvoiceTimestamp")).alias("Recency"),
        countDistinct("InvoiceNo").alias("Frequency"),
        sum("TotalSales").alias("Monetary")
    )
    
    # 3. Define Window specs for 5-quintile scoring (1-5)
    # Recency: Lower is better, so we order ASC
    recency_window = Window.orderBy(col("Recency").asc())
    # Frequency & Monetary: Higher is better, so we order DESC
    frequency_window = Window.orderBy(col("Frequency").desc())
    monetary_window = Window.orderBy(col("Monetary").desc())

    # 4. Calculate R, F, M scores
    rfm_scores = rfm_df.withColumn("R_Score", ntile(5).over(recency_window)) \
                       .withColumn("F_Score", ntile(5).over(frequency_window)) \
                       .withColumn("M_Score", ntile(5).over(monetary_window))

    # 5. Concatenate scores to create a segment
    rfm_final = rfm_scores.withColumn(
        "RFM_Segment",
        col("R_Score").cast("string") + col("F_Score").cast("string") + col("M_Score").cast("string")
    )

    save_output(rfm_final, "rfm_segmentation")

# --- ANALYSIS 2 & 3: AOV & CLV ---
def analyze_aov_clv(df):
    """
    Calculates Average Order Value (AOV) and Customer Lifetime Value (CLV).
    """
    print("Starting AOV & CLV Analysis...")
    
    customer_summary = df.groupBy("CustomerID", "Country").agg(
        sum("TotalSales").alias("CLV"),  # CLV is just total sales per customer
        countDistinct("InvoiceNo").alias("TotalOrders")
    )
    
    # AOV = Total Sales / Total Orders
    aov_clv_df = customer_summary.withColumn(
        "AOV",
        col("CLV") / col("TotalOrders")
    )

    save_output(aov_clv_df, "aov_clv_summary")

# --- ANALYSIS 4: Product Affinity ---
def analyze_product_affinity(df):
    """
    Finds pairs of products (StockCode) frequently bought together.
    This uses a self-join.
    """
    print("Starting Product Affinity Analysis...")
    
    # 1. We only need InvoiceNo and StockCode.
    invoice_items = df.select("InvoiceNo", "StockCode").distinct()
    
    # 2. Self-join the dataframe on InvoiceNo
    # We create two "aliases" (views) of the same data
    df_a = invoice_items.alias("a")
    df_b = invoice_items.alias("b")
    
    # Join condition:
    # 1. Same InvoiceNo (they were in the same basket)
    # 2. Item A < Item B (avoids duplicates like (B,A) and self-pairs like (A,A))
    joined_df = df_a.join(
        df_b,
        (col("a.InvoiceNo") == col("b.InvoiceNo")) & \
        (col("a.StockCode") < col("b.StockCode")),
        "inner"
    )
    
    # 3. Count pairs and order by most frequent
    affinity_df = joined_df.groupBy(
        col("a.StockCode").alias("Item_A"),
        col("b.StockCode").alias("Item_B")
    ).agg(
        count("*").alias("PurchaseCount")
    ).orderBy(col("PurchaseCount").desc())

    save_output(affinity_df, "product_affinity")

# --- ANALYSIS 5: Return Rate ---
def analyze_return_rate(df):
    """
    Calculates return rates for products.
    NOTE: This function MUST receive the *unfiltered* DataFrame,
    as it needs to see negative quantities.
    """
    print("Starting Return Rate Analysis...")
    
    # 1. Use conditional aggregation
    rate_df = df.groupBy("StockCode", "Description").agg(
        sum(when(col("Quantity") > 0, col("Quantity")).otherwise(0)).alias("TotalSold"),
        abs(sum(when(col("Quantity") < 0, col("Quantity")).otherwise(0))).alias("TotalReturned")
    )
    
    # 2. Calculate rate. Add 1 to TotalSold to avoid divide-by-zero errors.
    rate_df = rate_df.withColumn(
        "ReturnRate",
        col("TotalReturned") / (col("TotalSold") + 1)
    )
    
    # 3. Filter for noise (e.g., only show items sold > 100 times)
    rate_df_filtered = rate_df.where("TotalSold > 100") \
                              .orderBy(col("ReturnRate").desc())

    save_output(rate_df_filtered, "return_rates")

# --- ANALYSIS 6: Peak Times ---
def analyze_peak_times(df):
    """
    Finds peak purchase hours and days of the week.
    """
    print("Starting Peak Time Analysis...")
    
    # 1. Extract hour and day of week
    time_df = df.withColumn("Hour", hour(col("InvoiceTimestamp"))) \
                .withColumn("DayOfWeek", dayofweek(col("InvoiceTimestamp")))
    
    # 2. Aggregate by Hour
    peak_hour_df = time_df.groupBy("Hour") \
                          .agg(countDistinct("InvoiceNo").alias("TotalOrders")) \
                          .orderBy("Hour")
                          
    # 3. Aggregate by DayOfWeek
    # (Note: 1=Sunday, 2=Monday... 7=Saturday in Spark SQL)
    peak_day_df = time_df.groupBy("DayOfWeek") \
                         .agg(countDistinct("InvoiceNo").alias("TotalOrders")) \
                         .orderBy("DayOfWeek")

    save_output(peak_hour_df, "peak_purchase_hours")
    save_output(peak_day_df, "peak_purchase_days")

# --- ANALYSIS 7: Purchase Cadence ---
def analyze_purchase_cadence(df):
    """
    Finds the average time (in days) between purchases for each customer.
    This uses a Window function (lag).
    """
    print("Starting Purchase Cadence Analysis...")
    
    # 1. We only need customer and *unique* invoice dates
    customer_dates = df.select("CustomerID", "InvoiceTimestamp").distinct()
    
    # 2. Define a window partitioned by customer, ordered by date
    window_spec = Window.partitionBy("CustomerID").orderBy("InvoiceTimestamp")
    
    # 3. Use lag() to find the *previous* purchase date
    df_with_lag = customer_dates.withColumn(
        "PrevPurchaseDate",
        lag("InvoiceTimestamp", 1).over(window_spec)
    )
    
    # 4. Calculate days between purchases
    df_with_days = df_with_lag.withColumn(
        "DaysBetween",
        datediff(col("InvoiceTimestamp"), col("PrevPurchaseDate"))
    ).dropna(subset=["DaysBetween"]) # Drop the first purchase (null)
    
    # 5. Find the average cadence per customer
    cadence_df = df_with_days.groupBy("CustomerID") \
                             .agg(avg("DaysBetween").alias("AvgPurchaseCadence_Days")) \
                             .orderBy("AvgPurchaseCadence_Days")

    save_output(cadence_df, "purchase_cadence")

# --- MAIN EXECUTION ---
def main():
    """
    Main entry point for the Spark job.
    """
    print("--- Starting Big Data Analytics Job ---")
    
    spark = SparkSession.builder \
        .appName("ECommerce_Analytics") \
        .getOrCreate()
    
    # Set log level to WARN to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    try:
        # 1. Load Data
        print(f"Loading data from {HDFS_INPUT_PATH}...")
        base_df = spark.read.parquet(HDFS_INPUT_PATH)

        # 2. --- PRE-PROCESSING ---
        print("Starting data pre-processing...")
        
        # Convert InvoiceDate to a proper timestamp
        # The format "M/d/yyyy H:mm" is common for this dataset
        df_with_dates = base_df.withColumn(
            "InvoiceTimestamp",
            to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm")
        )
        
        # Calculate TotalSales
        df_with_sales = df_with_dates.withColumn(
            "TotalSales",
            col("Quantity") * col("UnitPrice")
        )

        # --- Create two base DataFrames for analysis ---

        # 1. Base for most analyses: Filtered for valid orders
        #    - No null CustomerID (can't analyze behavior without a customer)
        #    - Positive quantity (purchases, not returns)
        #    - Positive unit price (not free items)
        analysis_df = df_with_sales.dropna(subset=["CustomerID"]) \
                                   .filter(col("Quantity") > 0) \
                                   .filter(col("UnitPrice") > 0)

        # 2. Base for Return Rate: Needs negative quantities
        return_analysis_df = df_with_sales.dropna(subset=["StockCode"])
        
        print(f"Pre-processing complete. Caching analysis_df for performance...")
        analysis_df.cache() # Cache for faster re-use

        # 3. --- RUN ALL ANALYSES ---
        analyze_rfm(analysis_df)
        analyze_aov_clv(analysis_df)
        analyze_product_affinity(analysis_df)
        analyze_return_rate(return_analysis_df) # Use the unfiltered DF
        analyze_peak_times(analysis_df)
        analyze_purchase_cadence(analysis_df)

        print("--- All analyses complete. ---")

    except Exception as e:
        print(f"\n!!! AN ERROR OCCURRED: {e} !!!\n")
        print("Stopping Spark job with error.")
    
    finally:
        # 4. Stop Spark Session
        print("Stopping Spark session.")
        spark.stop()

if __name__ == "__main__":
    main()