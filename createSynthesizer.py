import pandas as pd
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import os
from sdv.single_table import CTGANSynthesizer
import matplotlib.pyplot as plt

# Load CSVs from the folder
data = pd.read_csv('files/test_measure_simplified.csv')

# Automatically detect metadata from the dataframe
# metadata = Metadata.detect_from_dataframe(
#     data=data,
#     table_name='test_measure_simplified'
# )

# graph = metadata.visualize()
# graph.format = 'png'  # Set the output format to PNG
# graph.render(filename='metadata_visualization', cleanup=True)  # Saves and removes .dot file

# Save the detected metadata to a file
metadata_path = "test_measure_simplified_metadata.json"

# metadata.save_to_json(metadata_path)
print(f"Metadata detected and saved to {metadata_path}")



try:
    metadata = Metadata.load_from_json(metadata_path)
    metadata.validate()
    print("Metadata is valid!")
except ValueError as e:
    print(f"Metadata validation failed: {e}")

synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=True,
    epochs=500,
    verbose=True
)

synthesizer.fit(data)

synthesizer.save(
    filepath='my_CTsynthesizer.pkl'
)


synthetic_data = synthesizer.sample(num_rows=500)

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
plt.savefig("plot.png")  # Save the plot as a PNG
plt.close()

#fig = synthesizer.get_loss_values_plot()
#fig.show()





