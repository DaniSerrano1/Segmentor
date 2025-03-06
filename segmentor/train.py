# python train.py --workspace_path "/ruta/a/tu/workspace" --num_epochs 20 --batch_size 4 --learning_rate 0.0001 --validation --resume_checkpoint "/ruta/al/checkpoint.pth"
# ex: python train.py --workspace_path "/Users/danielserranodominguez/Segmentor"

import os
import argparse
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
    parser = argparse.ArgumentParser(description="Entrenamiento de YOLOLineDetector")
    parser.add_argument("--workspace_path", type=str, default="/Users/danielserranodominguez/Segmentor", help="Ruta del workspace")
    parser.add_argument("--num_epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=2, help="Tamaño del batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--poly_degree", type=int, default=3, help="Grado del polinomio")
    parser.add_argument("--grid_size", type=int, default=7, help="Tamaño de la grid")
    parser.add_argument("--B", type=int, default=2, help="Número de bounding boxes por celda")
    parser.add_argument("--lambdas", nargs=5, type=float, default=[1, 1, 0.5, 0.0001, 1], help="Ponderaciones de la loss")
    parser.add_argument("--validation", action="store_true", help="Activar validación")
    parser.add_argument("--val_interval", type=int, default=2, help="Intervalo de validación")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Checkpoint para reanudar entrenamiento")
    args = parser.parse_args()

    WORKSPACE_PATH = args.workspace_path
    LOGGER_PATH = os.path.join(WORKSPACE_PATH, "logs")
    images_path = os.path.join(WORKSPACE_PATH, "Curvelanes/valid/images")
    annotations_path = os.path.join(WORKSPACE_PATH, "Curvelanes/valid/annots")
    model_path = os.path.join(WORKSPACE_PATH, "checkpoints")
    os.makedirs(model_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    writer = SummaryWriter(log_dir=LOGGER_PATH)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((854, 480))
    ])

    print("\033[95mCreando dataset...\033[0m")
    dataset = LineDataset(annotations_path, images_path, grid_size=args.grid_size, poly_degree=args.poly_degree, B=args.B, transform=transform)
    print("Dataset creado!\n")

    print("\033[95mCreando dataloader...\033[0m")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("Dataloader creado!\n")

    print("\033[92mEjemplo de batch:\033[0m")
    # Obtener un batch de datos
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Mostrar el ejemplo de batch
    print("Tamaño del batch de imágenes:", images.shape)
    print("Tamaño del batch de etiquetas:", labels.shape, "\n")
    
    print("\033[93mCreando modelo...\033[0m")
    model = YOLOLineDetector(grid_size=args.grid_size, poly_degree=args.poly_degree, B=args.B).to(device)

    if args.resume_checkpoint:
        print(f"Cargando checkpoint desde {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = checkpoint['epoch'] + 1 if args.resume_checkpoint else 0

    print("\033[93mEntrenando el modelo...\033[0m")
    best_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} -> Procesando batches: "):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = total_loss(preds, labels, args.poly_degree, args.lambdas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_loss = total_train_loss / len(dataloader)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, os.path.join(model_path, "last_model.pth"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(model_path, "best_model.pth"))
            print(f"Nuevo mejor modelo guardado con pérdida: {best_loss:.4f}")

        if args.validation and (epoch + 1) % args.val_interval == 0:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images)
                    loss = total_loss(preds, labels, args.poly_degree, args.lambdas)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(dataloader)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            print(f"\033[94mValidación - Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_val_loss:.4f}\033[0m")

    print("\033[92mEntrenamiento completado!\033[0m")
    writer.close()