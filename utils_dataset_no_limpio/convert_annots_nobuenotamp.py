import json
import os
import numpy as np
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def fit_polynomial(points, degree, frame):
    """Ajusta un polinomio de grado 'degree' a los puntos (x, y) y dibuja en la imagen con OpenCV."""
    
    # Convertir a arrays de NumPy
    x = np.array([p[0] for p in points]).reshape(-1, 1)  
    y = np.array([p[1] for p in points]).reshape(-1, 1) 

    model = LinearRegression()
    
    if degree == 1:
        # Regresión lineal simple
        x_poly = x  
    else:
        # Ajuste polinómico (grado especificado)
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x)

    # Ajustar el modelo
    model.fit(x_poly, y)

    # Obtener coeficientes del modelo
    if degree == 1:
        c = model.coef_[0, 0]  # Pendiente
        d = model.intercept_[0]  # Intersección
        a, b = 0, 0  # No hay términos cúbicos ni cuadráticos
    else:
        a, b, c, d = model.coef_[0, -1], model.coef_[0, -2], model.coef_[0, -3], model.intercept_[0]

    # Definir rango de predicción
    x_min, x_max = int(min(x).item()), int(max(x).item())
    dy = int((max(y) - min(y)).item())  # Diferencia de altura de la línea

    # Predicción para graficar
    x_pred = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    if degree == 1:
        y_pred = model.predict(x_pred)
    else:
        y_pred = model.predict(poly.transform(x_pred))

    # Convertir y_pred a enteros para dibujar en la imagen
    x_pred = x_pred.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)

    # Dibujar los puntos originales en la imagen
    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), 5)

    # Dibujar la línea de regresión en la imagen
    for i in range(len(x_pred) - 1):
        pt1 = (x_pred[i], y_pred[i])
        pt2 = (x_pred[i + 1], y_pred[i + 1])
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # Azul

    return a, b, c, d, x_min, x_max, dy, frame  # Devolver la imagen con la regresión dibujada

def image_processing(image_path, annotation_path, grid_size=7, degree=3, B=2):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    with open(annotation_path, "r") as f:
        label_data = json.load(f)
        
    for line in label_data["Lines"]:
        # print(len(line))
        points = [(float(p["x"]), float(p["y"])) for p in line]
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        
        if len(line) > 2:
            a, b, c, d, x_min, x_max, dy, image = fit_polynomial(points, degree, image)
        else:
            a, b, c, d, x_min, x_max, dy, image = fit_polynomial(points, 1, image)

        # print(a, b, c, d, x_min, x_max, dy)
    
    return image

if __name__ == "__main__":
    annotations_path = "/Users/danielserranodominguez/Desktop/Segmentor/Curvelanes/valid/labels"
    images_path = "/Users/danielserranodominguez/Desktop/Segmentor/Curvelanes/valid/images"

    i = 0

    for annot_name in os.listdir(annotations_path):
        annotation_path = os.path.join(annotations_path, annot_name)
        image_path = annotation_path.replace("labels", "images").replace(".lines.json", ".jpg")

        image = image_processing(image_path, annotation_path)
        if i == 5:
            break
        
        if save:
            cv2.imwrite(image_path.replace("images", "output"), image)
        
        if view:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Mostrar imagen en Jupyter
            plt.figure(figsize=(10, 6))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.show()