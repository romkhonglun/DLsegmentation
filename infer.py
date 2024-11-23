import torch
from torch import optim
from collections import OrderedDict
import segmentation_models_pytorch as smp
import argparse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
args = parser.parse_args()

# Model setup
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
learning_rate = 0.001
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Load checkpoint
checkpoint = torch.load("model.pth")
optimizer.load_state_dict(checkpoint['optimizer'])

new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
# Load params
model.load_state_dict(new_state_dict)
model.eval()

# Use the image_path argument
image_path = args.image_path
print(f"Image path provided: {image_path}")

# Load the image
image = Image.open(image_path)

# Define the transformations
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Apply the transformations to the image
image_np = np.array(image)
transformed = transform(image=image_np)
input_tensor = transformed['image'].unsqueeze(0)  # Create a mini-batch as expected by the model

# Move the input to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Process the output (example: convert to numpy array and display)
output = output.squeeze().cpu().numpy()

# Convert the output to RGB format
output = np.transpose(output, (1, 2, 0))  # Change the shape to (H, W, C)
output = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
output = (output * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

# Display the input and output images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("Input Image")
axes[0].axis('off')

axes[1].imshow(output)
axes[1].set_title("Inference Output")
axes[1].axis('off')

plt.show()
