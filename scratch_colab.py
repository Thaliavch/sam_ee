import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

# Define individual layers as TensorDictModules
conv1 = TensorDictModule(
    nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1),  # 6 bands as input channels
    in_keys=[f"B{i+1}" for i in range(6)],  # Example band names B1, B2, ..., B6
    out_keys=["conv1_out"]
)

relu1 = TensorDictModule(
    nn.ReLU(),
    in_keys=["conv1_out"],
    out_keys=["relu1_out"]
)

conv2 = TensorDictModule(
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    in_keys=["relu1_out"],
    out_keys=["conv2_out"]
)

# Fully connected layer after flattening
flatten_and_fc = TensorDictModule(
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 10)  # Adjust based on spatial dimensions and output classes
    ),
    in_keys=["conv2_out"],
    out_keys=["output"]
)

# Chain modules in TensorDictSequential
model = TensorDictSequential(conv1, relu1, conv2, flatten_and_fc)
 #***********************************************
# Example from Colab to load data
import pandas as pd
import torch

# Assuming TRAIN_FILE_PATH contains the CSV training data
df_train = pd.read_csv(TRAIN_FILE_PATH)

# Split into features and labels
features = df_train[BANDS].values  # BANDS is a list of band column names, e.g., ["B1", "B2", ..., "B6"]
target = df_train[OUTPUT_CLASS].values  # OUTPUT_CLASS is the column name for labels

# Convert to PyTorch tensors
features = torch.tensor(features, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.long)

# Reshape features to [batch_size, num_bands, height, width]
num_samples, num_bands = features.shape[0], len(BANDS)
height, width = INPUT_TILE_X_DIM, INPUT_TILE_Y_DIM  # Define these dimensions based on your input tile size
reshaped_features = features.reshape(num_samples, num_bands, height, width)

# Create TensorDict for features
feature_dict = {f"B{i+1}": reshaped_features[:, i, :, :].unsqueeze(1) for i in range(num_bands)}
input_data = TensorDict(feature_dict, batch_size=[num_samples])
#********************************************************
from torch.utils.data import DataLoader, TensorDataset

def collate_fn(batch):
    # Separate inputs and targets, then stack them
    inputs, targets = zip(*batch)
    batched_inputs = torch.stack(inputs, dim=0)
    batched_targets = torch.stack(targets, dim=0)
    return batched_inputs, batched_targets

# Combine TensorDict with targets in TensorDataset
train_dataset = TensorDataset(input_data, target)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
#*********************************************************
# Define criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 13
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Convert inputs to TensorDict format
        inputs = TensorDict(inputs, batch_size=[inputs.size(0)])

        # Forward pass
        output_tensordict = model(inputs)
        output = output_tensordict["output"]

        # Calculate loss
        loss = criterion(output, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss every other epoch
    if (epoch) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

