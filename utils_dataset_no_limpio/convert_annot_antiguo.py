import os
import json
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

def draw_polynomial_line(image, param, color=(255, 0, 0), thickness=2):
    """
    Dibuja una línea polinómica sobre la imagen, donde los parámetros contienen los coeficientes del polinomio
    y el rango de valores de x (xmin, xmax) sobre los que se evalúa el polinomio.
    :param image: La imagen sobre la que se dibujará la línea.
    :param param: Lista de parámetros [coeff2, coeff1, coeff0, xmin, xmax, dy, conf].
    :param color: Color de la línea (por defecto verde).
    :param thickness: Grosor de la línea.
    :return: La imagen con la línea dibujada.
    """
    
    coeff5, coeff4, coeff3, coeff2, coeff1, coeff0, xmin, xmax, dy = param
    
    xmin = xmin
    xmax = xmax
    
    # Generar puntos (x, y) a partir del polinomio para el rango de x entre xmin y xmax
    points = []
    for x in np.linspace(xmin, xmax, 100):  # Usamos 100 puntos para suavizar la curva
        y = coeff5 * x**5 + coeff4 * x**4 + coeff3 * x**3 + coeff2 * x**2 + coeff1 * x + coeff0  # Ecuación del polinomio
        points.append((int(x), int(y)))
    print(f"points: {points}")
    AA
    #print(f"y = {coeff5} * x**5 + {coeff4} * x**4 + {coeff3} * x**3 + {coeff2} * x**2 + {coeff1} * x + {coeff0}")
    # Dibujar la línea en la imagen
    for i in range(1, len(points)):
        cv2.line(image, points[i-1], points[i], color, thickness)
    
    return image

def fit_polynomial_ridge(points, degree, alpha=0.01):
    """Ajusta un polinomio usando regresión Ridge para evitar sobreajuste."""
    points = sorted(points, key=lambda p: p[1])
    x = np.array([p[0] for p in points]).reshape(-1, 1)
    y = np.array([p[1] for p in points]).reshape(-1, 1)

    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(y)

    model = Ridge(alpha=alpha)  # Regularización para evitar sobreajuste
    model.fit(X_poly, x)

    coeffs = model.coef_.flatten().tolist()
    coeffs[0] = model.intercept_[0]  # Ajustar el término independiente

    x_min, x_max = np.min(x), np.max(x)
    dy = y[-1] - y[0]

    return coeffs, x_min, x_max, dy

def fit_polynomial(points, degree):
    """Ajusta un polinomio de grado 'degree' a los puntos (x, y)."""
    points = sorted(points, key=lambda p: p[1])  # Ordenar por y ascendente
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    if len(x) > degree:
        coeffs = np.polyfit(y, x, degree)  # Ajuste polinómico x = f(y)
    elif len(x) == 2:
        # Handle the case of a straight line (2 points)
        coeffs = np.polyfit(y, x, 1)  # Fit as a line (degree 1)
        coeffs = np.pad(coeffs, (degree + 1 - len(coeffs), 0), mode="constant")
    else:
        coeffs = np.zeros(degree + 1)  # Si no hay suficientes puntos

    x_min, x_max = min(x), max(x)
    
    # assert x_max <=2560 & x_min >= 0, "Invalid x_max or x_min"
    dy = (y[-1] - y[0])  # Diferencia de altura de la línea

    return coeffs.tolist(), x_min, x_max, dy

def process_curve_lanes(dataset_path, output_dir, grid_size=7, degree=2, B=2):
    """
    Convierte CurveLanes en el formato adecuado y guarda en una nueva carpeta.
    """
    label_paths = glob(os.path.join(dataset_path, "labels", "*.json"))
    converted_label_dir = os.path.join(dataset_path, output_dir)

    # Crear carpeta de salida si no existe
    os.makedirs(converted_label_dir, exist_ok=True)

    for label_path in tqdm(label_paths, desc="Processing labels"):
        with open(label_path, "r") as f:
            label_data = json.load(f)

        img_name = os.path.basename(label_path).replace(".lines.json", ".jpg")
        img_path = os.path.join(dataset_path, "images", img_name)
        
        # Cargar imagen para obtener tamaño
        image = cv2.imread(img_path)
        h, w, _ = image.shape

        # Inicializar matriz de etiquetas
        label_matrix = np.zeros((grid_size, grid_size, B * (degree + 5)))

        for line in label_data["Lines"]:
            points = [(float(p["x"]), float(p["y"])) for p in line]
            coeffs, x_min, x_max, dy = fit_polynomial(points, degree)

            # Asegurarse que los coeficientes tienen tamaño degree+1
            coeffs = np.array(coeffs)

            # No normalizar los coeficientes
            # coeffs = coeffs / max(abs(c) for c in coeffs)  # Ya no lo normalizamos

            x_min, x_max, dy = x_min / w, x_max / w, dy / h  # Escalar valores

            # Determinar celda en la cuadrícula
            grid_x = int((x_min + x_max) / 2 * grid_size)
            grid_y = int(dy * grid_size)

            # Evitar valores fuera de rango
            grid_x = min(max(grid_x, 0), grid_size - 1)
            grid_y = min(max(grid_y, 0), grid_size - 1)

            # Buscar un espacio en B
            for b in range(B):
                start_idx = b * (degree + 5)
                if label_matrix[grid_y, grid_x, start_idx + degree + 4] == 0:
                    label_matrix[grid_y, grid_x, start_idx:start_idx + degree + 5] = \
                        np.concatenate([coeffs, [x_min, x_max, dy, 1.0]])
                    break

        # Guardar el archivo de anotación convertido
        output_label_path = os.path.join(converted_label_dir, os.path.basename(label_path))
        with open(output_label_path, "w") as f:
            json.dump({"label": label_matrix.tolist()}, f, indent=4)

# Convertir los conjuntos de datos y guardar en una nueva carpeta
# process_curve_lanes("/Users/danielserranodominguez/Desktop/Segmentor/Curvelanes/valid", "converted_labels")

annotation_path = "/Users/danielserranodominguez/Desktop/Segmentor/Curvelanes/valid/labels/0a0d988cfa3c35dc6a7f90135c591148.lines.json"
image_path = "/Users/danielserranodominguez/Desktop/Segmentor/Curvelanes/valid/images/0a0d988cfa3c35dc6a7f90135c591148.jpg"
#output_image_path = "/Users/danielserranodominguez/Desktop/Segmentor/outputs/output_image_with_annotations_0a0d988cfa3c35dc6a7f90135c591148.jpg"

def intent(image_path, annotation_path, grid_size=7, degree=5, B=2):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    with open(annotation_path, "r") as f:
        label_data = json.load(f)
        
    label_matrix = np.zeros((grid_size, grid_size, B * (degree + 5)))
    for line in label_data["Lines"]:
        print(len(line))
        points = [(float(p["x"]), float(p["y"])) for p in line]
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        cv2.imwrite("points.jpg", image)
        
        coeffs, x_min, x_max, dy = fit_polynomial_ridge(points, degree)
                
        params = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], x_min, x_max, dy

        image = draw_polynomial_line(image, params)
        
        cv2.imwrite("line.jpg", image)
        
        

intent(image_path, annotation_path)