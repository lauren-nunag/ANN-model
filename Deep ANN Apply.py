'''
Student ANN Apply code
'''

# Packages
import torch
import torch.nn as nn
import pandas as pd  # For reading Excel files

# Hyperparameters
num_input_units = 1  # Direct input units (no flattening)
num_output_units = 3  # Direct output units
num_hidden_layers = 5  # Number of hidden layers
num_units_per_hidden_layer = 8  # Number of units in each hidden layer
output_activation = 'linear'  # Set this to 'linear'

# Color codes for terminal output
WHITE = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'

# File paths
excel_file = '../ANN-model/files/Train Validate Apply Data Synthesized.xlsx'  # Path to the Excel file containing training, validation, and apply data
apply_input_sheet = 'Apply Inputs'  # Sheet name for apply input patterns
model_path = f'trained_ANN_{num_input_units}x{num_units_per_hidden_layer}x{num_hidden_layers}.pth'  # Path to load the trained ANN model

# Custom dataset class for Excel input/output (with sheet names and skipping the first row of titles)
class ApplyDataset:
    def __init__(self, excel_file, input_sheet):
        # Skip the first row (titles) by setting skiprows=1
        self.input_patterns = pd.read_excel(excel_file, sheet_name=input_sheet, header=None, skiprows=1).values
        self.num_patterns = len(self.input_patterns)
        print(f'{BLUE}| Applying the ANN to {GREEN}{self.num_patterns} {BLUE}input patterns |{WHITE}')
    
    def __len__(self):
        return self.num_patterns
    
    def __getitem__(self, idx):
        input_pattern = torch.tensor(self.input_patterns[idx], dtype=torch.float32)
        return input_pattern

# Model definition (Fully Connected)
class FramePredictor(nn.Module):
    def __init__(self, num_input_units, num_output_units, num_hidden_layers, num_units_per_hidden_layer, output_activation='linear'):
        super(FramePredictor, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(num_input_units, num_units_per_hidden_layer))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_units_per_hidden_layer, num_units_per_hidden_layer))
            layers.append(nn.ReLU())
        
        # Output layer (linear activation)
        layers.append(nn.Linear(num_units_per_hidden_layer, num_output_units))
        
        # Store the layers in a Sequential container
        self.ANN_layers = nn.Sequential(*layers)
        
        # Using Identity for linear output (no activation)
        if output_activation == 'linear':
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation function: {output_activation}")
        
    def forward(self, x):
        # Input already one-dimensional, no need to flatten
        x = self.ANN_layers(x)
        x = self.output_activation(x)  # Apply the linear output activation
        return x  # Output remains (batch_size, num_output_units)

# Load the apply dataset
apply_dataset = ApplyDataset(excel_file, apply_input_sheet)

# Load the trained model
model = FramePredictor(num_input_units, num_output_units, num_hidden_layers, num_units_per_hidden_layer, output_activation=output_activation)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model state dict with weights_only set to True
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Applying the model to the apply dataset
apply_results_file = 'apply_results.txt'
with open(apply_results_file, 'w') as results_file:
    # Add header to the output file
    results_file.write("ANN Output\n")  # Header added here
    for idx in range(len(apply_dataset)):
        input_pattern = apply_dataset[idx].to(device)
        with torch.no_grad():
            output = model(input_pattern.unsqueeze(0))  # Add batch dimension
        output_np = output.cpu().numpy().flatten()
        # Write the results to file (comma-separated)
        results_file.write(','.join(map(str, output_np)) + '\n')

print(f'{BLUE}| Results saved to {GREEN}{apply_results_file} {BLUE}|{WHITE}')
