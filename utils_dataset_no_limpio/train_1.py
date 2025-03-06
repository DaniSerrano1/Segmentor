import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import LineDataset
from model import YOLOLineDetector
from loss import total_loss
from tqdm import tqdm

if __name__ == "__main__":
    WORKSPACE_PATH = "/Users/danielserranodominguez/Segmentor"
    images_path = os.path.join(WORKSPACE_PATH, "Curvelanes/valid/images")
    annotations_path = os.path.join(WORKSPACE_PATH, "Curvelanes/output/annots")
    
    # Hiperparámetros
    num_epochs = 10
    batch_size = 2
    learning_rate = 1e-4
    poly_degree = 3  # Ajusta según la configuración de tu modelo
    grid_size = 7  # Mismo grid_size que usaste en el modelo
    B = 2  # Número de bounding boxes por celda
    lambdas = [1, 1, 0.5, 0.0001, 1]  # Ponderaciones de la loss

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # Transformaciones
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize((640, 640))
        transforms.Resize((854, 480))
    ])

    # Dataloader
    print("\033[95mCreando dataset...\033[0m\n")
    print("poly_degree: ", poly_degree)
    print("grid_size: ", grid_size)
    print("B: ", B)
    dataset = LineDataset(annotations_path, images_path, grid_size=grid_size, poly_degree=poly_degree, B=B, transform=transform)
    print("Dataset creado!\n")
    
    print("\033[95mCreando dataloader...\033[0m")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataloader creado!\n")

    print("\033[92mEjemplo de batch:\033[0m")
    # Obtener un batch de datos
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Mostrar el ejemplo de batch
    print("Tamaño del batch de imágenes:", images.shape)
    print("Tamaño del batch de etiquetas:", labels.shape, "\n")
    
    print("\033[93mCreando modelo...\033[0m")
    model = YOLOLineDetector(grid_size=grid_size, poly_degree=poly_degree, B=B).to(device)
    print("Modelo creado!\n")
    
    print("\033[93mEntrenando el modelo...\033[0m")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\033[92mHiperparámetros:\033[0m")
    print("Num epochs: ", num_epochs)
    print("Batch_size: ", batch_size)
    print("Optimizer: ", optimizer.__class__.__name__)
    print("Learning rate: ", learning_rate)
    print("Lambdas: ", lambdas)
    
    # Loop de entrenamiento
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} -> Procesando batches: "):
            images, labels = images.to(device), labels.to(device)

            # Forward
            preds = model(images)
            loss = total_loss(preds, labels, poly_degree, lambdas)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Promediar la pérdida de la época
        avg_loss = total_train_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("\033[92mEntrenamiento completado!\033[0m")
    

# print("\033[91mEste es un mensaje en rojo\033[0m")  # Rojo
# print("\033[92mEste es un mensaje en verde\033[0m")  # Verde
# print("\033[93mEste es un mensaje en amarillo\033[0m")  # Amarillo
# print("\033[94mEste es un mensaje en azul\033[0m")  # Azul
# print("\033[95mEste es un mensaje en magenta\033[0m")  # Magenta
# print("\033[96mEste es un mensaje en cian\033[0m")  # Cian
# print("\033[97mEste es un mensaje en blanco\033[0m")  # Blanco