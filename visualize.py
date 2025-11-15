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
    
    top_15_pairs = df.nlargest(15, 'PurchaseCount')
    top_15_pairs['Pair'] = top_15_pairs['Item_A'] + "  +  " + top_15_pairs['Item_B']
    
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
    plot_return_rate(return_df)
    plot_peak_times(hour_df, day_df)
    plot_purchase_cadence(cadence_df)
    
    print("--- Visualization Script Complete ---")
    print(f"All plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()