import pandas as pd

# Define the filenames
RAW_DATA_FILE = 'original_maize_data.csv'
CLEAN_DATA_FILE = 'maize_clean_for_training.csv'

# These are the exact column names from your CMS screenshot
RATE_COLUMN = 'रेट /kg'
DATE_COLUMN = 'Date'

def clean_data():
    print(f"Starting data cleaning... loading '{RAW_DATA_FILE}'")
    
    try:
        df = pd.read_csv(RAW_DATA_FILE)
    except FileNotFoundError:
        print(f"---")
        print(f"ERROR: File not found: '{RAW_DATA_FILE}'")
        print("Please export your data from the CMS and save it in this folder.")
        print(f"---")
        return

    # 1. Check if essential columns exist
    if RATE_COLUMN not in df.columns or DATE_COLUMN not in df.columns:
        print(f"---")
        print(f"ERROR: Columns not found. Expected '{RATE_COLUMN}' and '{DATE_COLUMN}'.")
        print(f"Found columns: {df.columns.to_list()}")
        print(f"---")
        return

    # 2. Clean the 'रेट /kg' (Rate/kg) column
    # This function splits "20.50 - 22.50" into low/high and finds the average
    def calculate_average_price(rate_string):
        if not isinstance(rate_string, str):
            return None
        
        parts = rate_string.strip().split('-')
        
        try:
            low = float(parts[0].strip())
            high = float(parts[1].strip()) if len(parts) > 1 else low
            return (low + high) / 2
        except (ValueError, TypeError):
            return None

    df['avg_price'] = df[RATE_COLUMN].apply(calculate_average_price)

    # 3. Clean the 'Date' column
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')

    # 4. Drop rows with missing crucial data
    df = df.dropna(subset=[DATE_COLUMN, 'avg_price'])

    # 5. Rename columns for Prophet
    # Prophet requires 'ds' for the date and 'y' for the value
    df_prophet = df.rename(columns={
        DATE_COLUMN: 'ds',
        'avg_price': 'y'
    })

    # 6. Select *only* the columns we need
    df_clean = df_prophet[['ds', 'y']]

    # 7. Sort by date, which is required for time-series
    df_clean = df_clean.sort_values(by='ds')
    
    if df_clean.empty:
        print("---")
        print("ERROR: No data was left after cleaning. Please check your raw CSV file.")
        print("---")
        return

    # 8. Save the final clean file
    df_clean.to_csv(CLEAN_DATA_FILE, index=False)

    print(f"---")
    print(f"Cleaning Complete! Processed {len(df_clean)} rows.")
    print(f"Clean data saved to '{CLEAN_DATA_FILE}'")
    print(f"---")

if __name__ == "__main__":
    clean_data()