import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io

class SAMSegmentationHandler(BaseHandler):
    """
    Custom handler to process 3-band RGB images, apply SAM for segmentation,
    and return segmentation mask.
    """
    def initialize(self, context):
        # Load the SAM model
        properties = context.system_properties
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")
        
        # Load SAM model
        model_type = "vit_b"  # Use desired model type (e.g., vit_h for higher capacity)
        checkpoint_path = f"{model_dir}/sam_checkpoint.pth"  # Path to SAM checkpoint file
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(self.device).eval()
        
        # Initialize the SAM predictor
        self.predictor = SamPredictor(self.sam)

    def preprocess(self, data):
        # Extract image from data (assuming a 3-band image is sent as bytes)
        image_bytes = data[0].get("body")
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to numpy array and ensure it's in RGB format
        rgb_image = np.array(image.convert("RGB"))
        
        # Send the image to the SAM predictor
        self.predictor.set_image(rgb_image)
        return rgb_image

    def inference(self, rgb_image):
        # Define point prompt for SAM segmentation (example point provided)
        input_point = np.array([[100, 100]])  # Adjust as needed for your segmentation
        input_label = np.array([1])  # 1 for foreground
        
        # Predict segmentation mask
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        
        return masks[0]  # Return the first mask

    def postprocess(self, masks):
        # Convert mask to a format suitable for returning (e.g., binary image in bytes)
        mask_image = Image.fromarray((masks * 255).astype(np.uint8))
        mask_bytes = io.BytesIO()
        mask_image.save(mask_bytes, format="PNG")
        return [mask_bytes.getvalue()]

###############################################################################

import torch
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model (replace "vit_b" with the appropriate model type if different)
model_type = "vit_b"
checkpoint_path = "path/to/checkpoint.pth"  # Replace with your SAM checkpoint path
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

# Set SAM model to evaluation mode and move it to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
sam.eval()

# Initialize the predictor
predictor = SamPredictor(sam)

# Set the RGB image for prediction
predictor.set_image(rgb_image)

