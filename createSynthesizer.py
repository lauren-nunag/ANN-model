from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import os
from sdv.single_table import CTGANSynthesizer

# Load CSVs from the folder
datasets = load_csvs(
    folder_name="C:\\Users\\jspag\\PycharmProjects\\CTGANSynthesizer\\files\\",
    read_csv_parameters={
        'skipinitialspace': True,
        'encoding': 'utf_8'
    }
)

data = datasets['test_measure']

# Keep only the HR column
data = data[['HR']]  # Subset to include only the HR column

# Automatically detect metadata from the dataframe
# metadata = Metadata.detect_from_dataframe(
#     data=data,
#     table_name='test_measure'
# )

# Save the detected metadata to a file
metadata_path = "HR_Only_path_to_metadata.json"

#metadata.save_to_json(metadata_path)
#print(f"Metadata detected and saved to {metadata_path}")



try:
    metadata = Metadata.load_from_json(metadata_path)
    metadata.validate()
    print("Metadata is valid!")
except ValueError as e:
    print(f"Metadata validation failed: {e}")

synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=False,
    epochs=500,
    verbose=True
)

synthesizer.fit(data)

synthesizer.save(
    filepath='my_synthesizer.pkl'
)

fig = synthesizer.get_loss_values_plot()
fig.show()





