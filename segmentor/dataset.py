import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from tqdm import tqdm

class LineDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, grid_size=7, poly_degree=3, B=2, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.grid_size = grid_size
        self.poly_degree = poly_degree
        self.B = B

        self.image_files = [img_name.split('.')[0] for img_name in os.listdir(img_dir) if img_name.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        img_path = f"{self.img_dir}/{image_name}.jpg"
        annotation_path = f"{self.annotations_dir}/{image_name}.txt"

        # Leer imagen
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        if self.transform:
            image = self.transform(image)

        # Inicializar la matriz de etiquetas
        label = torch.zeros(self.grid_size, self.grid_size, self.B * (self.poly_degree + 4))  # +4 para los 4 coeficientes + x_min, x_max y la confianza

        # Leer el archivo de anotaciones
        with open(annotation_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Extraer los valores de cada línea
            values = line.strip().split()
            coeffs = [float(v) for v in values[:4]]  # a, b, c, d
            x_min, x_max = int(values[4]), int(values[5])

            # Normalización a [0,1]
            x_min, x_max = x_min / w, x_max / w
            coeffs = torch.tensor(coeffs, dtype=torch.float32)

            # Calcular la ubicación en la cuadrícula
            grid_x = int(x_min * self.grid_size)
            grid_y = 0  # Solo se usa grid_x, no grid_y

            # Buscar un espacio vacío en B
            for b in range(self.B):
                if label[grid_y, grid_x, b * (self.poly_degree + 4) + self.poly_degree + 3] == 0:  # La posición de confianza es la última
                    label[grid_y, grid_x, b * (self.poly_degree + 4):(b + 1) * (self.poly_degree + 4)] = \
                        torch.cat([coeffs, torch.tensor([x_min, x_max, 1.0], dtype=torch.float32)])  # Añadir los coeficientes, x_min, x_max y confianza
                    break

        return image, label



# class LineDataset(Dataset):
#     def __init__(self, annotations_dir, img_dir, grid_size=7, poly_degree=3, B=2, transform=None, output_txt="valid_images.txt"):
#         self.img_dir = img_dir
#         self.annotations_dir = annotations_dir
#         self.transform = transform
#         self.grid_size = grid_size
#         self.poly_degree = poly_degree
#         self.B = B
#         self.output_txt = output_txt

#         self.image_files = []

#         # Verificar si el archivo .txt ya existe
#         if os.path.exists(self.output_txt):
#             print("Reading image names from file...")
#             # Leer los nombres de las imágenes desde el archivo
#             with open(self.output_txt, "r") as txt_file:
#                 self.image_files = [line.strip() for line in txt_file.readlines()]
#         else:
#             # Filtrar imágenes y escribir en el archivo .txt si no existe
#             with open(self.output_txt, "w") as txt_file:
#                 for img_name in tqdm(os.listdir(img_dir), desc="Filtering images"):
#                     img_path = os.path.join(img_dir, img_name)
#                     image = cv2.imread(img_path)
#                     if image is not None and image.shape == (1440, 2560, 3):  # Verificar la forma de la imagen
#                         # Guardar solo las imágenes que cumplen con el tamaño
#                         annotation_path = f"{self.annotations_dir}/{img_name.split('.')[0]}.txt"
#                         if os.path.exists(annotation_path):
#                             self.image_files.append(img_name.split('.')[0])
#                             txt_file.write(img_name.split('.')[0] + "\n")  # Escribir en el archivo .txt

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         image_name = self.image_files[idx]
#         img_path = f"{self.img_dir}/{image_name}.jpg"
#         annotation_path = f"{self.annotations_dir}/{image_name}.txt"
        
#         # Leer imagen
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         h, w, _ = image.shape

#         if self.transform:
#             image = self.transform(image)

#         # Inicializar la matriz de etiquetas
#         label = torch.zeros(self.grid_size, self.grid_size, self.B * (self.poly_degree + 5))  # +5 para los 4 coeficientes + x_min, x_max, dy y la confianza

#         # Leer el archivo de anotaciones
#         with open(annotation_path, "r") as f:
#             lines = f.readlines()

#         for line in lines:
#             # Extraer los valores de cada línea
#             values = line.strip().split()
#             coeffs = [float(v) for v in values[:4]]  # a, b, c, d
#             x_min, x_max, dy = int(values[4]), int(values[5]), int(values[6])

#             # # Normalización a [0,1]
#             x_min, x_max = x_min / w, x_max / w
#             coeffs = torch.tensor(coeffs, dtype=torch.float32)
#             # coeffs = torch.tensor(coeffs) / max(abs(c) for c in coeffs)  # Normalizamos los coeficientes
#             dy = dy / h

#             # Calcular la ubicación en la cuadrícula
#             grid_x = int(x_min * self.grid_size)
#             grid_y = int(dy * self.grid_size)

#             # Buscar un espacio vacío en B
#             for b in range(self.B):
#                 if label[grid_y, grid_x, b * (self.poly_degree + 5) + self.poly_degree + 4] == 0:  # La posición de confianza es la última
#                     label[grid_y, grid_x, b * (self.poly_degree + 5):(b + 1) * (self.poly_degree + 5)] = \
#                         torch.cat([coeffs, torch.tensor([x_min, x_max, dy, 1.0], dtype=torch.float32)])  # Añadir los coeficientes, x_min, x_max, dy y confianza
#                     break

#         return image, label