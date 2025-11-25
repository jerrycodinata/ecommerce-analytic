from pyspark.sql import SparkSession

# --- CONFIGURATION ---
CSV_SOURCE = "hdfs://namenode:9000/data/ecommerce/ecommerce.csv"
PARQUET_OUTPUT = "hdfs://namenode:9000/data/ecommerce/ecommerce.parquet"

def main():
    spark = SparkSession.builder \
        .appName("CSV_to_Parquet_Converter") \
        .getOrCreate()

    print(f"--- Reading CSV from {CSV_SOURCE} ---")

    # Read CSV
    # header=True: Uses the first row as column names
    # inferSchema=True: Automatically guesses if a column is int, string, etc.
    df = spark.read.option("header", "true") \
                   .option("inferSchema", "true") \
                   .csv(CSV_SOURCE)

    print("--- Schema Inferred: ---")
    df.printSchema()

    print(f"--- Writing Parquet to {PARQUET_OUTPUT} ---")

    df.write.mode("overwrite").parquet(PARQUET_OUTPUT)

    print("--- Conversion Complete! ---")
    spark.stop()

if __name__ == "__main__":
    main()