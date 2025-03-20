import torch
import json

# Load PyTorch quantized model
pt_model = torch.load("sample_data/resnet18_quantised.pt", map_location="cpu")

# Load JSON model summary
with open("sample_data/resnet18_quantized.json", "r") as f:
    json_model = json.load(f)

# Print PyTorch model state dictionary keys
print("\n--- PyTorch Model State Dict Keys ---")
print(pt_model.state_dict().keys())  # This should list all layer names

# Print first layer in JSON model for comparison
print("\n--- JSON Model First Layer ---")
first_layer_name = list(json_model.keys())[0]
print(first_layer_name, "->", json_model[first_layer_name])
