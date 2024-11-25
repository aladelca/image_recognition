import base64
import json
from utils import preprocess_image
import numpy as np
from PIL import Image
from model.neural_network import ImageRecognition
from model.utils import equivalences
import boto3
import torch


def lambda_handler(event, context):
    # Extract the Base64-encoded image
    encoded_image = event.get("image", None)
    
    if encoded_image:
        # Decode the image
        image_data = base64.b64decode(encoded_image)

        # Save or process the image
        with open("/tmp/uploaded_image.jpg", "wb") as image_file:
            image_file.write(image_data)
        image = Image.open(image_file)
        image_tensor = preprocess_image(image)
        if image_tensor.ndim == 3:
            # Add a batch dimension
            image_tensor  = np.expand_dims(image_tensor, axis=0)

        # Initialize the S3 client
        s3 = boto3.client("s3")
        local_file_path = "/tmp/model.pth"
        # Download the file from S3
        s3.download_file("image-recognition-players", "model/custom_vgg16.pth", local_file_path)
        model = ImageRecognition(num_classes=6)
        device = torch.device("cpu")
        # Predict
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).permute(0, 3, 1, 2)
        predicted_class = model.predict_from_load(image_tensor, 
                        local_file_path,
                        device)
        # Show prediction
        predicted_class = equivalences[int(predicted_class.item())]
        return predicted_class