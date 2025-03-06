import cv2
import numpy as np

# Lista de puntos
puntos = [(1286, 1885), (1294, 1900), (1303, 1916), (1312, 1931), (1320, 1947), (1329, 1962), 
          (1338, 1977), (1346, 1992), (1355, 2007), (1364, 2021), (1372, 2036), (1381, 2050), 
          (1390, 2065), (1399, 2079), (1407, 2093), (1416, 2107), (1425, 2121), (1433, 2135)]

# Crear un marco negro de 2560x1440
frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

# Dibujar la línea en el marco usando los puntos
for i in range(1, len(puntos)):
    cv2.line(frame, puntos[i-1], puntos[i], (0, 255, 0), 2)  # Línea verde, grosor 2

# Guardar la imagen
cv2.imwrite("linea_generada.jpg", frame)