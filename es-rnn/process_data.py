import os
import pandas as pd


def clean_data(csv_file_path, cleaned_csv_file_path):
    """
    Cleans the exchange rate data to have the same time series length for all currencies.

    Parameters:
    - csv_file_path (str): The path to the original CSV file.
    - cleaned_csv_file_path (str): The path to save the cleaned CSV file.
    """
    df         = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df         = df.sort_values('Date')

    drop_columns = []
    for col in df.columns:
        if df[col].isna().any():
            drop_columns.append(col)

    df.drop(columns=drop_columns, inplace=True)

    min_year = df['Date'].dt.year.min()
    max_year = df['Date'].dt.year.max()

    df = df[(df['Date'].dt.year >= min_year) & (df['Date'].dt.year <= max_year)]
    df.to_csv(cleaned_csv_file_path, index=False)

    print(f"Data cleaned and saved to {cleaned_csv_file_path}. Dropped columns: {drop_columns}")

def resample_data(input_csv_path, output_folder):
    """
    Resamples the dataset to different frequencies (daily, weekly, monthly, quarterly, yearly)
    and saves them into separate CSV files.

    Parameters:
    - input_csv_path (str): The path to the original CSV file.
    - output_folder (str): The folder where the resampled CSV files will be saved.
    """
    df         = pd.read_csv(input_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Resampling frequencies
    frequencies = {
        'daily':     'D',
        'weekly':    'W',
        'monthly':   'M',
        'quarterly': 'Q',
        'yearly':    'Y'
    }
    
    for freq_name, freq_code in frequencies.items():
        resampled_df = df.resample(freq_code).mean()
        resampled_df.dropna(how='all', inplace=True)
        
        output_csv_path = os.path.join(output_folder, f"{freq_name}.csv")
        resampled_df.to_csv(output_csv_path)
