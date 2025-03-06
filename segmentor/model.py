import torch
import torch.nn as nn

class YOLOLineDetector(nn.Module):
    def __init__(self, grid_size=7, poly_degree=3, B=2):
        super().__init__()
        self.grid_size = grid_size
        self.poly_degree = poly_degree
        self.B = B
        self.output_dim = B * (poly_degree + 4)  # (a, b, c, d, x_min, x_max, conf)

        # Backbone CNN para entrada (3, 854, 480)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (64, 427, 240)
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (128, 213, 120)
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (256, 106, 60)
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (512, 53, 30)
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)   # (1024, 26, 15)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(1024 * 13 * 7, 1024), nn.ReLU(),
            nn.Linear(1024, grid_size * grid_size * self.output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)  # (batch, 1024, 26, 15)
        x = x.view(x.shape[0], -1)  # (batch, 1024 * 26 * 15)
        x = self.fc(x)  # (batch, grid_size * grid_size * output_dim)
        x = x.view(-1, self.grid_size, self.grid_size, self.output_dim)
        return x

# class YOLOLineDetector(nn.Module):
#     def __init__(self, grid_size=7, poly_degree=3, B=2):
#         super().__init__()
#         self.grid_size = grid_size
#         self.poly_degree = poly_degree
#         self.B = B
#         self.output_dim = B * (poly_degree + 5)

#         # Backbone CNN ajustada para entrada (3, 640, 640)
#         self.backbone = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # (64, 320, 320)
#             nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # (128, 160, 160)
#             nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # (256, 80, 80)
#             nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # (512, 40, 40)
#             nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(2, 2)   # (1024, 20, 20)
#         )

#         # Fully connected layers optimizadas
#         self.fc = nn.Sequential(
#             nn.Linear(1024 * 10 * 10, 1024), nn.ReLU(),
#             nn.Linear(1024, grid_size * grid_size * self.output_dim)
#         )

#     def forward(self, x):
#         x = self.backbone(x)  # (batch, 1024, 20, 20)
#         x = x.view(x.shape[0], -1)  # Aplanar -> (batch, 1024*20*20)
#         x = self.fc(x)  # (batch, grid_size*grid_size*output_dim)
#         x = x.view(-1, self.grid_size, self.grid_size, self.output_dim)
#         return x
    

# # class YOLOLineDetector(nn.Module):
# #     def __init__(self, grid_size=7, poly_degree=3, B=2):
# #         super().__init__()
# #         self.grid_size = grid_size
# #         self.poly_degree = poly_degree
# #         self.B = B
# #         self.output_dim = B * (poly_degree + 5)  # Coeficientes + x_min, x_max, dy, p

# #         # Backbone CNN para imágenes de entrada (3, 1440, 2560)
# #         self.backbone = nn.Sequential(
# #             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
# #             nn.MaxPool2d(2, 2),  # (64, 720, 1280)
# #             nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
# #             nn.MaxPool2d(2, 2),  # (128, 360, 640)
# #             nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
# #             nn.MaxPool2d(2, 2),  # (256, 180, 320)
# #             nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
# #             nn.MaxPool2d(2, 2),  # (512, 90, 160)
# #             nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(),
# #             nn.MaxPool2d(2, 2)   # (1024, 45, 80)
# #         )

# #         # Fully connected layers para mapear a la salida
# #         self.fc = nn.Sequential(
# #             nn.Linear(1024 * 45 * 80, 1024), nn.ReLU(),
# #             nn.Linear(1024, grid_size * grid_size * self.output_dim)
# #         )

# #     def forward(self, x):
# #         x = self.backbone(x)
# #         x = x.view(x.shape[0], -1)  # Aplanar
# #         x = self.fc(x)
# #         x = x.view(-1, self.grid_size, self.grid_size, self.output_dim)
# #         return x
