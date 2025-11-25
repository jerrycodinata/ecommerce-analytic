# ECommerce Analytics

Lightweight Big Data analytics pipeline for the Online Retail dataset (UK retail sales). The project provides data cleaning, Spark-based analyses, and matplotlib/Seaborn visualizations.

Contents

- `data_cleaning.py` — Pandas-based cleaning and parquet export
- `scripts/analytic.py` — PySpark job that runs analyses and writes Parquet results to `scripts/results`
- `visualize.py` — Small plotting utility that reads results and saves figures to `scripts/visualizations`
- `convert-to-parquet.py` — helper to convert CSV to Parquet (Kaggle dataset helper)

Key analyses produced by `scripts/analytic.py`

- RFM segmentation (`rfm_segmentation`)
- AOV and CLV per customer (`aov_clv_summary`)
- Product affinity (co-purchased pairs) (`product_affinity`)
- Return rates (`return_rates`)
- Peak purchase hours and days (`peak_purchase_hours`, `peak_purchase_days`)
- Purchase cadence per customer (`purchase_cadence`)

Design notes

- The Spark job creates two analysis bases:
  - `analysis_df`: filtered sales rows (positive quantity, non-null CustomerID) used for most customer/product analyses.
  - `return_analysis_df`: keeps negative quantities (returns) but filters out known non-product SKUs and administrative/error descriptions (e.g., `wrongly coded`, `postage`, `bank charge`) so return-rate metrics are not skewed by fees or administrative rows.
- Product affinity output includes both `StockCode` and `Description` and a pre-computed `PairLabel` in the form `Description (StockCode) + Description (StockCode)` (truncated for plotting).

Quickstart — Local (development)

Prerequisites

- Python 3.8+ with: `pandas`, `pyarrow` or `fastparquet`, `matplotlib`, `seaborn`.
- Apache Spark (2.4+ or 3.x) if you want to run `scripts/analytic.py` with `spark-submit`.

Install (pip)

```bash
python -m pip install pandas pyarrow matplotlib seaborn
```

If you use Spark locally, ensure the `spark-submit` on your PATH points to a Spark distribution.

1. Set Docker Containers

```bash
docker compose up -d
```

2. Move Data to Hadoop

```bash
docker exec -it namenode bash
hdfs dfs -mkdir /data/ecommerce/
hdfs dfs -put /data/ecommerce.csv /data/ecommerce/
```

Convert CSV data to parquet

```bash
docker exec spark-master /spark/bin/spark-submit /opt/scripts/conversion.py
```

Run the PySpark job (example using local Spark):

```bash
docker exec spark-master /spark/bin/spark-submit /opt/scripts/analytic.py
```

Notes:

- `analytic.py` expects the input Parquet to be available at `HDFS_INPUT_PATH` (default points to `hdfs://namenode:9000/data/ecommerce/ecommerce.parquet`).
- Results are saved to `scripts/results/<analysis_name>` as a single Parquet part file.

4. Generate visualizations

After you have run the analytics job and the `scripts/results` directory contains the Parquet outputs, generate PNGs:

```bash
python visualize.py
```

Output images are written to `scripts/visualizations/`:

- `1_rfm_segment_distribution.png`
- `2a_aov_distribution.png`, `2b_clv_distribution.png`
- `3_product_affinity.png`, `7_product_affinity_top10.png`
- `4_return_rates.png`
- `5a_peak_hours.png`, `5b_peak_days.png`
- `6_purchase_cadence.png`
- `7_product_affinity_top10.png`

Development notes & customization

- To change truncation widths or label formatting for product affinity, edit `visualize.py`'s `_two_line_label` helper or alter the `label_len` variable in `scripts/analytic.py`.
- The return-rate preprocessing filters a set of known non-product `StockCode`s and description keywords. Extend `non_product_exact`, `non_product_prefixes`, or `admin_keywords_pattern` in `scripts/analytic.py` to tailor filtering to your dataset.
- If your environment uses HDFS, ensure the Spark job's `HDFS_INPUT_PATH` points to the correct HDFS location and that the runtime has access to `OUTPUT_DIR_BASE` for writing.

Troubleshooting

- If `visualize.py` fails to read a Parquet file, ensure the `INPUT_DIR` (`scripts/results`) contains a dataset folder with a single part Parquet file (Spark writes `_SUCCESS` and `part-*.snappy.parquet`).
- Common errors when running Spark:
  - `pyarrow` / parquet read errors: install `pyarrow` or `fastparquet` in the environment you use to run Python-based parts.
  - Permission errors writing to `OUTPUT_DIR_BASE`: update `OUTPUT_DIR_BASE` in `scripts/analytic.py` to a writable path inside your runtime container or host.

License & attribution

- This repository is a small internal analytics demo built for educational and exploratory purposes.
