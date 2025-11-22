import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_DIR = "./scripts/results"
OUTPUT_DIR = "./scripts/visualizations"
SNS_PALETTE = "viridis"

# --- HELPER FUNCTIONS ---

def load_data(name):
    """Loads a Parquet file from the results directory."""
    path = os.path.join(INPUT_DIR, name)
    print(f"Loading data from {path}...")
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        print(f"!!! Error loading {path}: {e}")
        return None

def save_plot(fig, name):
    """Saves a Matplotlib figure to the visualizations directory."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    print(f"Plot saved to {path}")
    plt.close(fig) # Close the figure to save memory

# --- PLOTTING FUNCTIONS ---

def plot_rfm(df):
    """Plots the top 15 RFM segments."""
    if df is None: return
    
    # Get the top 15 segments by customer count
    segment_counts = df['RFM_Segment'].value_counts().nlargest(15).reset_index()
    segment_counts.columns = ['RFM_Segment', 'CustomerCount']

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=segment_counts,
        y='RFM_Segment',
        x='CustomerCount',
        palette=SNS_PALETTE,
        ax=ax
    )
    ax.set_title("Top 15 RFM Segments by Customer Count", fontsize=16)
    ax.set_xlabel("Number of Customers")
    ax.set_ylabel("RFM Segment (R-F-M)")
    save_plot(fig, "1_rfm_segment_distribution.png")

def plot_aov_clv(df):
    """Plots the distribution of AOV and CLV."""
    if df is None: return
    
    # Plot AOV Distribution (filtering extreme outliers)
    aov_filtered = df[df['AOV'].between(0, 1000)]['AOV']
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(aov_filtered, bins=50, kde=True, ax=ax)
    ax.set_title("Average Order Value (AOV) Distribution (0-1000)", fontsize=16)
    ax.set_xlabel("AOV")
    ax.set_ylabel("Frequency")
    save_plot(fig, "2a_aov_distribution.png")
    
    # Plot CLV Distribution (filtering extreme outliers)
    clv_filtered = df[df['CLV'].between(0, 5000)]['CLV']
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(clv_filtered, bins=50, kde=True, ax=ax)
    ax.set_title("Customer Lifetime Value (CLV) Distribution (0-5000)", fontsize=16)
    ax.set_xlabel("CLV")
    ax.set_ylabel("Frequency")
    save_plot(fig, "2b_clv_distribution.png")

def plot_product_affinity(df):
    """Plots the top 15 most frequent product pairs."""
    if df is None: return
    top_15_pairs = df.nlargest(15, 'PurchaseCount').copy()

    # Prefer the pre-computed PairLabel (created by analytic.py). If missing,
    # fall back to descriptions or stock codes depending on what's present.
    if 'PairLabel' in top_15_pairs.columns:
        top_15_pairs['Pair'] = top_15_pairs['PairLabel']
    else:
        # If descriptions exist, use them (fall back to codes).
        if 'Item_A_Desc' in top_15_pairs.columns and 'Item_B_Desc' in top_15_pairs.columns:
            top_15_pairs['Pair'] = (
                top_15_pairs['Item_A_Desc'].fillna(top_15_pairs.get('Item_A', top_15_pairs.get('Item_A_Code', ''))).astype(str)
                + "  +  " +
                top_15_pairs['Item_B_Desc'].fillna(top_15_pairs.get('Item_B', top_15_pairs.get('Item_B_Code', ''))).astype(str)
            )
        else:
            # Final fallback: try Item_A/Item_B or Item_A_Code/Item_B_Code
            a_col = 'Item_A' if 'Item_A' in top_15_pairs.columns else ('Item_A_Code' if 'Item_A_Code' in top_15_pairs.columns else None)
            b_col = 'Item_B' if 'Item_B' in top_15_pairs.columns else ('Item_B_Code' if 'Item_B_Code' in top_15_pairs.columns else None)
            if a_col and b_col:
                top_15_pairs['Pair'] = top_15_pairs[a_col].astype(str) + "  +  " + top_15_pairs[b_col].astype(str)
            else:
                # As a last resort, stringify the index
                top_15_pairs['Pair'] = top_15_pairs.index.astype(str)

    # Ensure pair labels are compact and rendered as two lines: left item on line 1,
    # '+ ' + right item on line 2. Truncate each side to keep the plot readable.
    def _two_line_label(s, left_width=40, right_width=40):
        if not isinstance(s, str):
            s = str(s)
        # Try common separators
        sep = None
        for candidate in ['\n+\n', '\n+ ', '  +  ', ' + ', '+']:
            if candidate in s:
                sep = candidate
                break
        if sep is None:
            # No explicit separator: just truncate and return two-line string
            left = s
            right = ''
        else:
            left, right = s.split(sep, 1)
        left = left.strip()
        right = right.strip()
        if len(left) > left_width:
            left = left[:left_width-3] + '...'
        if len(right) > right_width:
            right = right[:right_width-3] + '...'
        if right:
            return f"{left}\n+ {right}"
        else:
            return left

    top_15_pairs['Pair'] = top_15_pairs['Pair'].apply(lambda s: _two_line_label(s))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=top_15_pairs,
        y='Pair',
        x='PurchaseCount',
        palette=SNS_PALETTE,
        ax=ax
    )
    ax.set_title("Top 15 Co-Purchased Product Pairs", fontsize=16)
    ax.set_xlabel("Frequency (Times Purchased Together)")
    ax.set_ylabel("Product Pair")
    save_plot(fig, "3_product_affinity.png")


def plot_product_affinity_top10(df):
    """Plots the top 10 most frequent product pairs in a compact vertical layout."""
    if df is None: return

    top_10 = df.nlargest(10, 'PurchaseCount').copy()

    # Use PairLabel if present, otherwise fall back to constructed pair text
    if 'PairLabel' in top_10.columns:
        top_10['Pair'] = top_10['PairLabel']
    else:
        if 'Item_A_Desc' in top_10.columns and 'Item_B_Desc' in top_10.columns:
            top_10['Pair'] = (
                top_10['Item_A_Desc'].fillna(top_10.get('Item_A', top_10.get('Item_A_Code', ''))).astype(str)
                + "  +  " +
                top_10['Item_B_Desc'].fillna(top_10.get('Item_B', top_10.get('Item_B_Code', ''))).astype(str)
            )
        else:
            a_col = 'Item_A' if 'Item_A' in top_10.columns else ('Item_A_Code' if 'Item_A_Code' in top_10.columns else None)
            b_col = 'Item_B' if 'Item_B' in top_10.columns else ('Item_B_Code' if 'Item_B_Code' in top_10.columns else None)
            if a_col and b_col:
                top_10['Pair'] = top_10[a_col].astype(str) + "  +  " + top_10[b_col].astype(str)
            else:
                top_10['Pair'] = top_10.index.astype(str)

    # Two-line formatting helper (same logic as main affinity plot)
    def _two_line_label(s, left_width=36, right_width=36):
        if not isinstance(s, str):
            s = str(s)
        sep = None
        for candidate in ['\n+\n', '\n+ ', '  +  ', ' + ', '+']:
            if candidate in s:
                sep = candidate
                break
        if sep is None:
            left = s
            right = ''
        else:
            left, right = s.split(sep, 1)
        left = left.strip()
        right = right.strip()
        if len(left) > left_width:
            left = left[:left_width-3] + '...'
        if len(right) > right_width:
            right = right[:right_width-3] + '...'
        if right:
            return f"{left}\n+ {right}"
        else:
            return left

    top_10['Pair'] = top_10['Pair'].apply(lambda s: _two_line_label(s))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_10,
        y='Pair',
        x='PurchaseCount',
        palette=SNS_PALETTE,
        ax=ax
    )
    ax.set_title("Top 10 Co-Purchased Product Pairs", fontsize=14)
    ax.set_xlabel("Frequency (Times Purchased Together)")
    ax.set_ylabel("Product Pair")
    save_plot(fig, "7_product_affinity_top10.png")

def plot_return_rate(df):
    """Plots the top 15 products with the highest return rates."""
    if df is None: return
    
    # Filter for products with a meaningful number of sales
    df_filtered = df[df['TotalSold'] > 100]
    top_15_returns = df_filtered.nlargest(15, 'ReturnRate')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=top_15_returns,
        y='Description',
        x='ReturnRate',
        palette='Reds_r',
        ax=ax
    )
    ax.set_title("Top 15 Products by Return Rate (Min. 100 Sales)", fontsize=16)
    ax.set_xlabel("Return Rate (Returns / Sales)")
    ax.set_ylabel("Product Description")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    save_plot(fig, "4_return_rates.png")

def plot_peak_times(hour_df, day_df):
    """Plots peak purchase hours and days."""
    if hour_df is None or day_df is None: return
    
    # Plot Peak Hours
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=hour_df,
        x='Hour',
        y='TotalOrders',
        marker='o',
        ax=ax
    )
    ax.set_title("Total Orders by Hour of Day", fontsize=16)
    ax.set_xlabel("Hour (24-Hour Clock)")
    ax.set_ylabel("Total Orders")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(range(0, 24))
    save_plot(fig, "5a_peak_hours.png")

    # Plot Peak Days
    # Map Spark's 1-7 (Sun-Sat) to readable names
    day_map = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
    day_df['DayName'] = day_df['DayOfWeek'].map(day_map)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=day_df,
        x='DayName',
        y='TotalOrders',
        palette=SNS_PALETTE,
        order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        ax=ax
    )
    ax.set_title("Total Orders by Day of Week", fontsize=16)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Total Orders")
    save_plot(fig, "5b_peak_days.png")

def plot_purchase_cadence(df):
    """Plots the distribution of the average time between purchases."""
    if df is None: return
    
    # Filter for cadences < 90 days to see the main distribution
    cadence_filtered = df[df['AvgPurchaseCadence_Days'].between(1, 90)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(cadence_filtered['AvgPurchaseCadence_Days'], bins=45, kde=True, ax=ax)
    avg_cadence = cadence_filtered['AvgPurchaseCadence_Days'].mean()
    ax.axvline(avg_cadence, color='red', linestyle='--', label=f'Avg: {avg_cadence:.1f} days')
    ax.set_title("Distribution of Average Purchase Cadence (1-90 Days)", fontsize=16)
    ax.set_xlabel("Average Days Between Purchases")
    ax.set_ylabel("Number of Customers")
    ax.legend()
    save_plot(fig, "6_purchase_cadence.png")

# --- MAIN EXECUTION ---

def main():
    """Main function to load all data and generate all plots."""
    print("--- Starting Visualization Script ---")
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all datasets
    rfm_df = load_data("rfm_segmentation")
    aov_clv_df = load_data("aov_clv_summary")
    affinity_df = load_data("product_affinity")
    return_df = load_data("return_rates")
    hour_df = load_data("peak_purchase_hours")
    day_df = load_data("peak_purchase_days")
    cadence_df = load_data("purchase_cadence")
    
    # Generate all plots
    plot_rfm(rfm_df)
    plot_aov_clv(aov_clv_df)
    plot_product_affinity(affinity_df)
    plot_product_affinity_top10(affinity_df)
    plot_return_rate(return_df)
    plot_peak_times(hour_df, day_df)
    plot_purchase_cadence(cadence_df)
    
    print("--- Visualization Script Complete ---")
    print(f"All plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()