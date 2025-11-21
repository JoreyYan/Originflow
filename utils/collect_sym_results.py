import os
import json
import csv
import glob
import logging

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory to search within.
# Using os.path.expanduser to handle '~' if needed in the future.
BASE_DIR = "/home/junyu/project/sym/motif_sym/paper_design/c"

# Define the pattern to find all relevant JSON files.
# This pattern looks for:
# - Any folder starting with 'c' inside BASE_DIR
# - A 'result' subfolder
# - Any sub-subfolder within 'result'
# - The file ending with summary_confidences_0.json
SEARCH_PATTERN = os.path.join(BASE_DIR, "c*", "result", "*", "*summary_confidences_0.json")

# Define the output CSV file name.
OUTPUT_CSV_FILE = "confidence_summary.csv"

# --- Main Script Logic ---
def main():
    """
    Main function to find JSON files, extract data, and write to a CSV.
    """
    logging.info(f"Starting search in: {BASE_DIR}")
    
    # Find all files matching the pattern
    json_files = glob.glob(SEARCH_PATTERN)
    
    if not json_files:
        logging.warning("No 'summary_confidences_0.json' files found.")
        logging.warning(f"Please check if the path and pattern are correct: {SEARCH_PATTERN}")
        return

    logging.info(f"Found {len(json_files)} JSON files to process.")
    
    # A list to hold all the data we extract
    results_data = []

    # Process each file
    for file_path in json_files:
        try:
            # Extract the required folder names from the path
            path_parts = file_path.split(os.sep)
            
            # Expected structure: /.../c/c_folder/result/design_folder/summary.json
            # c_folder is at index -4, design_folder is at index -2
            c_folder_name = path_parts[-4]
            design_folder_name = path_parts[-2]

            # Read the JSON file content
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract ptm and iptm values. Use .get() to avoid errors if keys are missing.
            ptm_value = data.get('ptm', 'N/A')
            iptm_value = data.get('iptm', 'N/A')
            
            # Append the extracted information to our results list
            results_data.append({
                'c_folder': c_folder_name,
                'design_folder': design_folder_name,
                'ptm': ptm_value,
                'iptm': iptm_value
            })

        except (IOError, json.JSONDecodeError, IndexError) as e:
            logging.error(f"Could not process file {file_path}: {e}")
            continue # Skip to the next file

    # Write the collected data to a CSV file
    if not results_data:
        logging.info("No data was successfully extracted. CSV file will not be created.")
        return

    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='') as csvfile:
            # Define the header fields for the CSV
            fieldnames = ['c_folder', 'design_folder', 'ptm', 'iptm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header and the data
            writer.writeheader()
            writer.writerows(results_data)
        
        logging.info(f"Successfully created CSV file: {os.path.abspath(OUTPUT_CSV_FILE)}")

    except IOError as e:
        logging.error(f"Failed to write to CSV file {OUTPUT_CSV_FILE}: {e}")


if __name__ == "__main__":
    main()