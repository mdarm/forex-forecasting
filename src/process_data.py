import pandas as pd

def clean_data(csv_file_path, cleaned_csv_file_path):
    """
    Cleans the exchange rate data to have the same time series length for all currencies.

    Parameters:
    - csv_file_path (str): The path to the original CSV file.
    - cleaned_csv_file_path (str): The path to save the cleaned CSV file.
    """
    # Read the entire CSV file
    df = pd.read_csv(csv_file_path)

    # Convert the 'Date' column to a datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the dataframe by date
    df = df.sort_values('Date')

    # Find columns that have NaN values
    drop_columns = []
    for col in df.columns:
        if df[col].isna().any():
            drop_columns.append(col)

    # Drop the identified columns from the DataFrame
    df.drop(columns=drop_columns, inplace=True)

    # Identify the common time period for all remaining currencies
    min_year = df['Date'].dt.year.min()
    max_year = df['Date'].dt.year.max()

    # Trim the DataFrame to only include the common time period
    df = df[(df['Date'].dt.year >= min_year) & (df['Date'].dt.year <= max_year)]

    # Save the cleaned data to a new CSV file
    df.to_csv(cleaned_csv_file_path, index=False)

    print(f"Data cleaned and saved to {cleaned_csv_file_path}. Dropped columns: {drop_columns}")
