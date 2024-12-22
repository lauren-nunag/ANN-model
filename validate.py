import pandas
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_csv("../ANN-model/files/test_measure_simplified.csv")

data.to_pickle('test_measure_simplified.pkl')

metadata_path = "test_measure_simplified_metadata.json"

try:
    metadata = Metadata.load_from_json(metadata_path)
    metadata.validate()
    print("Metadata is valid!")
except ValueError as e:
    print(f"Metadata validation failed: {e}")

synthesizer = CTGANSynthesizer.load('my_CTsynthesizer.pkl')

synthetic_data = synthesizer.sample(num_rows=100000)

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




