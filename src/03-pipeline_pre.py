import pandas as pd
import json
import os

def load_data_from_csv(file_path: str) -> pd.DataFrame | None:
    """
    Loads data from a CSV file, removes:
    - Rows where 'question' starts with 'Câu hỏi Câu hỏi'
    - Rows where 'question' ends with '...'
    - Duplicate questions

    Returns only 'question' and 'answer' columns.
    """
    import pandas as pd
    import os

    if not os.path.exists(file_path):
        print(f"Error: The CSV file '{file_path}' was not found.")
        return None

    try:
        # Load CSV
        df = pd.read_csv(file_path, encoding='utf-8')

        # Normalize question column
        df['question'] = df['question'].astype(str).str.strip()

        # --- Áp dụng bộ lọc ---
        condition_not_starts = ~df['question'].str.startswith("Câu hỏi Câu hỏi")
        condition_not_ends = ~df['question'].str.endswith("...")

        # Kết hợp hai điều kiện
        filtered_df = df[condition_not_starts & condition_not_ends].copy()

        # Xoá các dòng bị trùng câu hỏi
        filtered_df = filtered_df.drop_duplicates(subset=['question'])

        # Giữ lại các cột cần thiết
        result_df = filtered_df[['question', 'answer']].copy()

        print(f"Original rows: {len(df)}, after filtering: {len(result_df)}")
        return result_df

    except Exception as e:
        print(f"An error occurred while processing the CSV: {e}")
        return None

def load_data_from_json(file_path: str) -> pd.DataFrame | None:
    """
    Loads data from a JSON file into a pandas DataFrame.
    
    This function loads all data without any filtering.

    Args:
        file_path (str): The full path to the input JSON file.

    Returns:
        pd.DataFrame | None: A DataFrame containing the data, or None if the file is not found or an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: The JSON file '{file_path}' was not found. Please check the file path.")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Successfully loaded {len(df)} rows from the JSON file.")
        return df
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")
        return None

def merge_dataframes(df_csv: pd.DataFrame, df_json_unfiltered: pd.DataFrame) -> pd.DataFrame:
    """
    Merges two DataFrames by concatenating them vertically.
    
    Args:
        df_csv (pd.DataFrame): Filtered DataFrame from the CSV file.
        df_json_unfiltered (pd.DataFrame): Unfiltered DataFrame from the JSON file.

    Returns:
        pd.DataFrame: A merged DataFrame.
    """
    if df_csv is None and df_json_unfiltered is None:
        print("Both DataFrames are empty. Returning an empty DataFrame.")
        return pd.DataFrame()
    
    if df_csv is None or df_csv.empty:
        print("CSV DataFrame is empty. Returning only the JSON data.")
        return df_json_unfiltered
    if df_json_unfiltered is None or df_json_unfiltered.empty:
        print("JSON DataFrame is empty. Returning only the filtered CSV data.")
        return df_csv[['question', 'answer']]

    merged_df = pd.concat([df_csv, df_json_unfiltered], ignore_index=True)
    return merged_df[['question', 'answer']]

def save_data_to_csv(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Saves a pandas DataFrame to a CSV file, creating the directory if it doesn't exist.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_file_path (str): The full path where the CSV file will be saved.
    """
    if df is None or df.empty:
        print("DataFrame is empty. No data to save.")
        return

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: '{output_dir}'")

    try:
        df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"Successfully saved {len(df)} rows to '{output_file_path}'")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

# ---
# Main execution block
# ---

def main():
    """
    Main function to orchestrate the data processing pipeline.
    It defines file paths, loads data from a CSV and a JSON file, filters the JSON data,
    merges the two datasets, and saves the final output to a new CSV.
    """
    # --- 1. DEFINE FILE PATHS RELATIVE TO THE PROJECT ROOT ---
    # The script is located in the 'src' directory.
    # We go up one level to reach the project root.
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the path to 'src'
    project_root = os.path.join(script_dir, '..')        # Goes up to 'UIT@PubHealthQA'
    
    # Define input and output paths based on the project structure
    # Input 1: CSV file from 'data/bronze'
    csv_input_path = os.path.join(project_root, 'data', 'bronze', 'raw_QAPair.csv')
    
    # Input 2: JSON file from 'data/gold'
    json_input_path = os.path.join(project_root, 'data', 'gold', 'question_answer_pairs.json') 
    
    # Output file: CSV in 'data/silver' with the specified name
    output_csv_path = os.path.join(project_root, 'data', 'silver', 'silver_data.csv')

    print("--- Starting data processing pipeline ---")
    print(f"Script location: {script_dir}")
    print(f"Looking for CSV input at: {csv_input_path}")
    print(f"Looking for JSON input at: {json_input_path}")
    print(f"Output will be saved to: {output_csv_path}")

    # --- 2. LOAD DATA ---
    # Load CSV data with both filters applied
    df_csv = load_data_from_csv(csv_input_path)
    if df_csv is None:
        print("Aborting the process due to a CSV file error.")
        return

    # Load JSON data without any filtering
    df_json = load_data_from_json(json_input_path)
    if df_json is None:
        print("Aborting the process due to a JSON file error.")
        return

    # --- 3. FILTER AND MERGE DATA ---
    # As per the request, JSON data is not filtered.
    df_json_unfiltered = df_json # Renaming for clarity
    print("\nJSON data is used without any filtering.")

    print("\nMerging filtered CSV data with unfiltered JSON data...")
    merged_df = merge_dataframes(df_csv, df_json_unfiltered)
    print(f"Total rows after merging: {len(merged_df)}")

    # --- 4. SAVE THE FINAL OUTPUT ---
    print("\nSaving merged data to the final CSV file...")
    save_data_to_csv(merged_df, output_csv_path)

    print("\n--- Pipeline completed successfully! ---")

if __name__ == "__main__":
    main()