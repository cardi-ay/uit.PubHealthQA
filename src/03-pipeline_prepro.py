import pandas as pd
import json
import os
from typing import List, Union

def load_data_from_csv(file_path: str) -> pd.DataFrame | None:
    """
    Loads data from a CSV file, removes:
    - Rows where 'question' starts with 'Câu hỏi Câu hỏi'
    - Rows where 'question' ends with '...'
    - Duplicate questions

    Returns only 'question' and 'answer' columns.
    """
    if not os.path.exists(file_path):
        print(f"Error: The CSV file '{file_path}' was not found.")
        return None

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df['question'] = df['question'].astype(str).str.strip()
        
        condition_not_starts = ~df['question'].str.startswith("Câu hỏi Câu hỏi")
        condition_not_ends = ~df['question'].str.endswith("...")

        filtered_df = df[condition_not_starts & condition_not_ends].copy()
        filtered_df = filtered_df.drop_duplicates(subset=['question'])

        result_df = filtered_df[['question', 'answer']].copy()
        print(f"Original CSV rows: {len(df)}, after filtering: {len(result_df)}")
        return result_df

    except Exception as e:
        print(f"An error occurred while processing the CSV: {e}")
        return None

def load_data_from_json_list(file_paths: List[str]) -> pd.DataFrame | None:
    """
    Loads and concatenates data from a list of JSON files into a single DataFrame.
    """
    all_json_data = []
    total_rows = 0

    if not file_paths:
        print("No JSON files provided. Skipping JSON data loading.")
        return None

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: The JSON file '{file_path}' was not found. Skipping this file.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_json_data.extend(data)
                print(f"Successfully loaded {len(data)} rows from '{file_path}'.")
                total_rows += len(data)
        except json.JSONDecodeError:
            print(f"Error: The file '{file_path}' is not a valid JSON file. Skipping.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred while reading '{file_path}': {e}. Skipping.")
            continue
    
    if not all_json_data:
        print("No data was loaded from any of the provided JSON files.")
        return None

    df = pd.DataFrame(all_json_data)
    print(f"Total rows loaded from all JSON files: {total_rows}")
    return df

def merge_dataframes(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges a list of DataFrames by concatenating them vertically.
    """
    # Filter out None and empty DataFrames
    valid_dfs = [df for df in df_list if df is not None and not df.empty]
    
    if not valid_dfs:
        print("All input DataFrames are empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    merged_df = pd.concat(valid_dfs, ignore_index=True)
    
    # Ensure the final DataFrame only has 'question' and 'answer' columns
    if 'question' in merged_df.columns and 'answer' in merged_df.columns:
        return merged_df[['question', 'answer']]
    else:
        print("Warning: Final merged DataFrame does not contain 'question' and 'answer' columns.")
        return merged_df

def save_data_to_csv(df: pd.DataFrame, output_file_path: str) -> None:
    """
    Saves a pandas DataFrame to a CSV file, creating the directory if it doesn't exist.
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

# --- New function to find JSON files ---
def find_json_files_in_directory(directory: str, file_names: List[str]) -> List[str]:
    """
    Constructs full file paths for a list of file names within a given directory.
    
    Args:
        directory (str): The directory to search in.
        file_names (List[str]): A list of JSON file names (e.g., ['file1.json', 'file2.json']).

    Returns:
        List[str]: A list of full paths to the specified JSON files.
    """
    full_paths = []
    for file_name in file_names:
        full_path = os.path.join(directory, file_name)
        full_paths.append(full_path)
    return full_paths

# ---
# Main execution block
# ---

def main():
    """
    Main function to orchestrate the data processing pipeline.
    It defines file paths, automatically loads data from specified JSON files in a directory,
    merges the datasets, and saves the final output to a new CSV.
    """
    # --- 1. DEFINE FILE PATHS ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..')

    # Define the input directory for JSON files
    json_input_dir = os.path.join(project_root, 'data', 'gold')

    # --- Sửa đổi ở đây: Thêm tên các file JSON vào đây ---
    # Chỉ cần thêm tên file vào list này là xong!
    json_file_names = ['question_answer_pairs.json','LLM_generated_question_answer_20250627_032104.json','LLM_generated_question_answer_20250627_201322.json']
    
    # Define default input paths
    csv_input_path = os.path.join(project_root, 'data', 'bronze', 'raw_QAPair.csv')
    output_csv_path = os.path.join(project_root, 'data', 'silver', 'silver_data.csv')

    print("--- Starting data processing pipeline ---")
    print(f"Looking for CSV input at: {csv_input_path}")
    print(f"Scanning for JSON files in directory: {json_input_dir}")
    print(f"Output will be saved to: {output_csv_path}")

    # --- 2. AUTOMATICALLY FIND JSON FILE PATHS ---
    # This automatically creates a list of full paths from the file names
    json_input_paths = find_json_files_in_directory(json_input_dir, json_file_names)
    print(f"Found the following JSON files to load: {json_input_paths}")
    
    # --- 3. LOAD DATA ---
    # Load CSV data with filtering
    df_csv = load_data_from_csv(csv_input_path)
    
    # Load data from the list of JSON files
    df_json_list = load_data_from_json_list(json_input_paths)

    if df_csv is None and df_json_list is None:
        print("\nAborting the process as no data was loaded from any source.")
        return

    # --- 4. MERGE DATA ---
    print("\nMerging filtered CSV data with all JSON data...")
    merged_df = merge_dataframes([df_csv, df_json_list])
    
    if merged_df.empty:
        print("Merged DataFrame is empty after merging. Nothing to save.")
        return
        
    print(f"Total rows after merging: {len(merged_df)}")

    # --- 5. SAVE THE FINAL OUTPUT ---
    print("\nSaving merged data to the final CSV file...")
    save_data_to_csv(merged_df, output_csv_path)

    print("\n--- Pipeline completed successfully! ---")

if __name__ == "__main__":
    main()