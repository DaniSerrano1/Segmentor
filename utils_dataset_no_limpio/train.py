import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from dataset import LineDataset
from model import YOLOLineDetector
from loss import total_loss
from tqdm import tqdm

if __name__ == "__main__":
    WORKSPACE_PATH = "/Users/danielserranodominguez/Segmentor"
    LOGGER_PATH = os.path.join(WORKSPACE_PATH, "logs")
    images_path = os.path.join(WORKSPACE_PATH, "Curvelanes/valid/images")
    annotations_path = os.path.join(WORKSPACE_PATH, "Curvelanes/valid/annots")
    model_path = os.path.join(WORKSPACE_PATH, "checkpoints")
    os.makedirs(model_path, exist_ok=True)

    # Hiperparámetros
    num_epochs = 10
    batch_size = 2
    learning_rate = 1e-4
    poly_degree = 3
    grid_size = 7
    B = 2
    lambdas = [1, 1, 0.5, 0.0001, 1]
    validation = False
    val_interval = 2  # Número de épocas entre cada validación
    resume_checkpoint = None  # Ruta para continuar entrenamiento si existe

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # Logger
    writer = SummaryWriter(log_dir=os.path.join(WORKSPACE_PATH, "logs"))

    # Transformaciones
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((854, 480))
    ])

    # Datasets y Dataloaders
    print("\033[95mCreando dataset...\033[0m")
    dataset = LineDataset(annotations_path, images_path, grid_size=grid_size, poly_degree=poly_degree, B=B, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataset creado!\n")

    # Modelo
    model = YOLOLineDetector(grid_size=grid_size, poly_degree=poly_degree, B=B).to(device)

    # Cargar checkpoint si se indica
    if resume_checkpoint:
        print(f"Cargando checkpoint desde {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if resume_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    # Loop de entrenamiento
    best_loss = float("inf")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} -> Procesando batches: "):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = total_loss(preds, labels, poly_degree, lambdas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_loss = total_train_loss / len(dataloader)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Guardar modelo cada época
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, os.path.join(model_path, "last_model.pth"))

        # Guardar el mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(model_path, "best_model.pth"))
            print(f"Nuevo mejor modelo guardado con pérdida: {best_loss:.4f}")

        if validation:
            # Validación
            if (epoch + 1) % val_interval == 0:
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for images, labels in dataloader:
                        images, labels = images.to(device), labels.to(device)
                        preds = model(images)
                        loss = total_loss(preds, labels, poly_degree, lambdas)
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(dataloader)
                writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
                print(f"\033[94mValidación - Epoch [{epoch+1}/{num_epochs}], Loss: {avg_val_loss:.4f}\033[0m")

    print("\033[92mEntrenamiento completado!\033[0m")
    writer.close()
