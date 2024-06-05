import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Define custom functions


def split_and_assign_headers(value):
    """
    Split value to extract headers.

    Parameters:
    - value (str): Input string.

    Returns:
    - str: Extracted header.
    """
    if not isinstance(value, str):
        return None, value

    # Replace unwanted characters with spaces
    value = re.sub(r'[^a-zA-Z0-9:. ]', ' ', value)

    # Split the value using colon (:) as a delimiter
    parts = value.split(":")

    # Return the first part as the header
    if len(parts) > 1:
        return parts[0].strip()
    else:
        return None, value.strip()


def split_and_assign_values(value):
    """
    Split value to extract values.

    Parameters:
    - value (str): Input string.

    Returns:
    - str: Extracted value.
    """
    if not isinstance(value, str):
        return None, value

    # Replace unwanted characters with spaces
    value = re.sub(r'[^a-zA-Z0-9:. ]', ' ', value)

    # Split the value using colon (:) as a delimiter
    parts = value.split(":")

    # Return the last part as the value
    if len(parts) > 1:
        return parts[-1].strip()
    else:
        return None, value.strip()


def rename_duplicate_columns(df):
    """
    Rename duplicate columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with renamed columns.
    """
    cols = pd.Series(df.columns)

    # Iterate over duplicated column names
    for dup in cols[cols.duplicated()].unique():
        # Rename duplicated columns with suffix '_i'
        cols[cols[cols == dup].index.values.tolist()] = \
            [dup + '_' + str(i) if i !=
             0 else dup for i in range(sum(cols == dup))]

    # Update DataFrame columns with renamed columns
    df.columns = cols

    return df


def clean_data(filename):
    """
    Clean and preprocess the data from a CSV file.

    Parameters:
    - filename (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Extract column names and reset the index
    column_names = df.columns.tolist()[:5]
    df = df.reset_index()

    # Rename columns with 'level_' prefix
    df = df.rename(columns=dict(
        zip(df.columns[:], ['level_{}'.format(i) for i in range(0, df.shape[1])])))

    # Rename the first 5 columns to original names
    df = df.rename(columns=dict(zip(df.columns[:5], column_names[:5])))

    # Identify object-type columns for header extraction
    object_columns = df.select_dtypes(include='object').columns

    # Copy DataFrame for header extraction
    df_headers = df.copy()

    # Extract headers using split_and_assign_headers function
    for col in object_columns[1:]:
        df_headers[col] = df_headers[col].apply(split_and_assign_headers)

    # Clean and format headers
    headers = df_headers.iloc[0, 5:].tolist()
    headers = [str(header).replace(' ', '_') if isinstance(
        header, (tuple, str)) else str(header) for header in headers]
    column_headers = df.columns[:5].tolist() + headers

    # Extract values using split_and_assign_values function
    for col in object_columns[1:]:
        df[col] = df[col].apply(split_and_assign_values)

    # Rename columns with cleaned headers
    df = df.rename(columns=dict(zip(df.columns[5:], column_headers[5:])))

    # Convert to numeric types (ignore errors)
    df = df.apply(pd.to_numeric, errors='ignore')

    # Convert 'ts' column to datetime
    df['ts'] = pd.to_datetime(df['ts'])

    # Handle duplicate columns by renaming them
    df = rename_duplicate_columns(df)

    return df


def merge_csv_files(directory):
    """
    Concatenates CSV files in a directory with the same column length.

    Parameters:
    - directory (str): The directory path containing CSV files.

    Returns:
    - pd.DataFrame: Concatenated DataFrame.
    """
    # List all CSV files in the specified directory
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Initialize an empty list to store DataFrames
    dfs = []

    # Iterate through each file
    for file in files:
        filename = os.path.join(directory, file)
        print(f"Processing file: {filename}")

        # Clean data using a custom function (assuming clean_data is defined)
        df = clean_data(filename=filename)

        # Append DataFrame to the list
        dfs.append(df)

    # Check if any DataFrames are present
    if not dfs:
        return None

    # Find common columns among all DataFrames
    common_columns = set(dfs[0].columns)
    print(f"Common Columns Set: {common_columns}")
    for df in dfs[1:]:
        common_columns = common_columns.intersection(df.columns)

    # Drop columns not in common_columns for each DataFrame
    for i in range(len(dfs)):
        dfs[i] = dfs[i][list(common_columns)]

    # Concatenate DataFrames only if they have the same columns
    col_len = len(common_columns)
    print(f"Column Length Set: {col_len}")
    for i in range(len(dfs)):
        if len(dfs[i].columns) != col_len:
            print(f"Columns not the same for file: {files[i]}")
            continue

    # Concatenate DataFrames
    merged_df = pd.concat(dfs, axis=0)

    # Drop columns with headers starting with '(None,'
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('None,')]

    merged_df.reset_index(drop=True, inplace=True)

    return merged_df
