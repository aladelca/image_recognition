import os
#os.chdir('../../src')
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from neural_network import ImageRecognition
from PIL import Image
from utils import equivalences, preprocess_image

# Streamlit UI
def main():
    st.title("Image Classification with PyTorch")
    st.write("Upload an image to get a prediction from the model.")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image_tensor = preprocess_image(image)
        if image_tensor.ndim == 3:
            # Add a batch dimension
            image_tensor  = np.expand_dims(image_tensor, axis=0)
        # Load the model
        model_path = "../../custom_vgg16.pth" 
        model = ImageRecognition(num_classes=6)
        #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device = torch.device("cpu")
        # Predict
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).permute(0, 3, 1, 2)
        predicted_class = model.predict_from_load(image_tensor, 
                        model_path,
                        device)
        # Show prediction
        predicted_class = equivalences[int(predicted_class.item())]

        st.write(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    main()
