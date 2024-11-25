import os
from PIL import Image
import numpy as np
def load_images_from_folder(folder_path, target_size=(256, 256), label = None):
    """
    Carga imágenes desde una carpeta, las redimensiona y las convierte en arrays.
    
    Args:
    - folder_path (str): Ruta de la carpeta que contiene las imágenes.
    - target_size (tuple): Tamaño al que se redimensionarán las imágenes (ancho, alto).

    Returns:
    - images_array (numpy.ndarray): Array de imágenes.
    - labels (list): Lista de nombres de archivo (opcionalmente puede ser la clase).
    """
    images_list = []  # Cambiamos el nombre a `images_list` para evitar confusión
    labels = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                # Redimensionar imagen al tamaño deseado
                img_resized = img.resize(target_size)
                # Convertir a array
                img_array = np.array(img_resized)
                # Asegurarse de que tiene 3 canales (RGB)
                if img_array.ndim == 2:  # Imagen en escala de grises
                    img_array = np.stack((img_array,) * 3, axis=-1)
                elif img_array.shape[-1] == 4:  # Imagen RGBA
                    img_array = img_array[..., :3]
                
                images_list.append(img_array)  # Agregar a la lista
                labels.append(label)  # Podrías extraer clases desde el nombre si es necesario
        except Exception as e:
            print(f"Error al cargar la imagen {file_path}: {e}")
    
    # Convertir la lista final a un array de NumPy
    images_array = np.array(images_list)
    return images_array, labels

def preprocess_image(img):
    img_resized = img.resize((256, 256))
        
    img_array = np.array(img_resized)
    if img_array.ndim == 2:  # Imagen en escala de grises
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # Imagen RGBA
        img_array = img_array[..., :3]
    return img_array/255

equivalences = {
    0: "cristiano ronaldo",
    1: 'lionel_messi',
    2: 'marco_reus',
    3: 'mbappe',
    4: 'neymar',
    5: 'robert_lewandowski'
}