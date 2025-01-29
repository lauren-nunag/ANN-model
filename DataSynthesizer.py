import math


from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def synthesize(rows):
    data = pd.read_pickle('test_measure_simplified.pkl')


    metadata_path = "test_measure_simplified_metadata.json"

    try:
        metadata = Metadata.load_from_json(metadata_path)
        metadata.validate()
        print("Metadata is valid!")
    except ValueError as e:
        print(f"Metadata validation failed: {e}")

    synthesizer = CTGANSynthesizer.load(
        filepath='my_CTsynthesizer.pkl'
    )

    synthetic_data = synthesizer.sample(num_rows=rows)

    # Define the target directory and base file name
    output_dir = r"C:\Users\jspag\PycharmProjects\syntheticModelTest\files"
    base_name = "CTGAN_synthetic_data"
    extension = ".xlsx"

    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing files and determine the next number
    counter = 1
    while os.path.exists(os.path.join(output_dir, f"{base_name}{counter}{extension}")):
        counter += 1

    # Construct the unique file name
    output_file_path = os.path.join(output_dir, f"{base_name}{counter}{extension}")

    # Save the file
    synthetic_data.to_excel(output_file_path, index=False)

    print(f"Synthetic data saved to {output_file_path}")

    diagnostic = run_diagnostic(
        real_data=data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )

    quality_report = evaluate_quality(
        data,
        synthetic_data,
        metadata
    )

    # Scatter plot for original data
    plt.scatter(data['Speed'], data['HR'], label='Original', alpha=0.5)

    # Scatter plot for synthetic data
    plt.scatter(synthetic_data['Speed'], synthetic_data['HR'], label='Synthetic', alpha=0.5)

    plt.xlabel('Speed')
    plt.ylabel('Heart Rate')
    plt.legend()
    plt.savefig("plot1.png")  # Save the plot as a PNG
    plt.close()

    excel_file = f"../ANN-model/files/{base_name}{counter}{extension}"
    cleanSheet(excel_file)

def cleanSheet(excel_file):
    data = pd.read_excel(excel_file)

    cleaned_data = data.dropna()
    cleaned_data.to_excel(excel_file, index=False)
    print(f"Rows with missing data removed. Updated data saved in {excel_file}")


def normalize_and_save_params(excel_file):
    """
    Normalize the data in an Excel file using simple scaling (by max value)
    and print the normalization parameters.
    """
    # Load the data
    data = pd.read_excel(excel_file, header=0)  # Assumes first row contains headers

    # Initialize dictionaries to store max values for inputs and outputs
    max_values = {}

    # Scale inputs (divide each column by its max value)
    scaled_data = data.copy()
    for column in data.columns:
        max_value = data[column].max()
        max_values[column] = max_value
        scaled_data[column] = data[column] / max_value

    # Print the scaling parameters (max values)
    print("Normalization parameters (max values for each column):")
    for column, max_value in max_values.items():
        print(f"{column}: {max_value}")

    # Save the normalized data to a new Excel file
    scaled_data.to_excel(excel_file, index=False)
    print(f"Normalized data saved to {'normalized_' + excel_file}.")

def splitSheet(excel_file):
    # Load the Excel file
    file_path = '../ANN-model/files/CTGAN_synthetic_data1.xlsx'  # Replace with your file path
    df = pd.read_excel(excel_file)

    # Define split indices
    total_rows = len(df)
    train_end = math.trunc((total_rows * .7)) # First 70% rows
    validate_end = train_end + math.trunc(total_rows * .2)  # Next 20% rows

    # Split the input column (first column)
    train_inputs = df.iloc[:train_end, [0]]
    validate_inputs = df.iloc[train_end:validate_end, [0]]
    apply_inputs = df.iloc[validate_end:, [0]]

    # Split the output columns (remaining 3 columns)
    train_outputs = df.iloc[:train_end, 1:]
    validate_outputs = df.iloc[train_end:validate_end, 1:]
    apply_outputs = df.iloc[validate_end:, 1:]

    # Save all splits to a single Excel file with six sheets
    output_file = '../ANN-model/files/Train Validate Apply Data Synthesized.xlsx'  # Name of the new Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        train_inputs.to_excel(writer, index=False, sheet_name='Train Inputs')
        validate_inputs.to_excel(writer, index=False, sheet_name='Validate Inputs')
        apply_inputs.to_excel(writer, index=False, sheet_name='Apply Inputs')
        train_outputs.to_excel(writer, index=False, sheet_name='Train Outputs')
        validate_outputs.to_excel(writer, index=False, sheet_name='Validate Outputs')
        apply_outputs.to_excel(writer, index=False, sheet_name='Apply Outputs')

    print(f"Data has been split and saved to {output_file}")

def merge_sheets(left_file, right_file, fkey):
    #Load the data
    Lfile = pd.read_csv(left_file)
    Rfile = pd.read_csv(right_file)

    #Merge using foreign key between files
    merged_data = pd.merge(Lfile, Rfile, on=fkey, how='inner')
    merged_data.to_csv('merged_data.csv', index=False)
    print("Merged dataset saved as 'merged_data.csv'")

#merge_sheets('files/test_measure.csv', 'files/subject-info.csv', 'ID_test')

#synthesize(10000)
#normalize_and_save_params('files/CTGAN_synthetic_data1.xlsx')
#splitSheet('../ANN-model/files/CTGAN_synthetic_data1.xlsx')