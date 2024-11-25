from neural_network import ImageRecognition
import torch
import os
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys

def load_images_from_folder(folder_path, target_size=(256, 256)):
    """
    Carga imágenes desde una carpeta, las redimensiona y las convierte en arrays.
    
    Args:
    - folder_path (str): Ruta de la carpeta que contiene las imágenes.
    - target_size (tuple): Tamaño al que se redimensionarán las imágenes (ancho, alto).

    Returns:
    - images_array (numpy.ndarray): Array de imágenes.
    - labels (list): Lista de nombres de archivo (opcionalmente puede ser la clase).
    """
    images_list = []  
    
    for filename in os.listdir(folder_path):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:

                img_resized = img.resize(target_size)

                img_array = np.array(img_resized)

                if img_array.ndim == 2:  
                    img_array = np.stack((img_array,) * 3, axis=-1)
                elif img_array.shape[-1] == 4:  
                    img_array = img_array[..., :3]
                
                images_list.append(img_array) 
            
        except Exception as e:
            print(f"Error al cargar la imagen {file_path}: {e}")
    
    # Convertir la lista final a un array de NumPy
    images_array = np.array(images_list)
    return images_array

class Predictor():

    def __init__(self):
        self.model = ImageRecognition(num_classes=6)

    def predict(self, x_test, weights):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        predicted_class = self.model.predict_from_load(x_test, weights, device)
        return predicted_class
    

def main():
    print(os.curdir)
    path = input()#sys.argv[1]
    
    predictor = Predictor()
    images = load_images_from_folder(path)
    x_test_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
    preds = predictor.predict(x_test_tensor, "../../custom_vgg16.pth")
    print(preds)
    return preds

if __name__ == "__main__":
    main()
