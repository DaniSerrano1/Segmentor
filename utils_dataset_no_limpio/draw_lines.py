import cv2
import numpy as np
import json

def draw_polynomial_line(image, param, color=(0, 255, 0), thickness=28):
    """
    Dibuja una línea polinómica sobre la imagen, donde los parámetros contienen los coeficientes del polinomio
    y el rango de valores de x (xmin, xmax) sobre los que se evalúa el polinomio.
    :param image: La imagen sobre la que se dibujará la línea.
    :param param: Lista de parámetros [coeff2, coeff1, coeff0, xmin, xmax, dy, conf].
    :param color: Color de la línea (por defecto verde).
    :param thickness: Grosor de la línea.
    :return: La imagen con la línea dibujada.
    """
    
    coeff2, coeff1, coeff0, xmin, xmax, dy, conf = param
    
    w = image.shape[1]
    xmin = xmin * w
    xmax = xmax * w
    print(f"{coeff2, coeff1, coeff0, xmin, xmax, dy, conf}")
    # Generar puntos (x, y) a partir del polinomio para el rango de x entre xmin y xmax
    points = []
    for x in np.linspace(xmin, xmax, 100):  # Usamos 100 puntos para suavizar la curva
        y = coeff2 * x**2 + coeff1 * x + coeff0  # Ecuación del polinomio
        points.append((int(x), int(y)))
    
    # Dibujar la línea en la imagen
    for i in range(1, len(points)):
        cv2.line(image, points[i-1], points[i], color, thickness)
    
    return image

def draw_original_lines(image, lines, color=(0, 0, 255), thickness=2):
    """
    Dibuja las líneas originales sobre la imagen usando las coordenadas (x, y).
    :param image: Imagen sobre la que se dibujará la línea.
    :param lines: Lista de líneas, cada línea es una lista de puntos (x, y).
    :param color: Color de la línea (por defecto rojo).
    :param thickness: Grosor de la línea.
    :return: Imagen con las líneas originales.
    """
    for line in lines:
        points = [(int(float(p["x"])), int(float(p["y"]))) for p in line]  # Asegurarse de convertir correctamente
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], color, thickness)
    return image

def plot_annotations_on_image(image_path, annotations_path, output_image_path, original_annotations_path=None, grid_size=7):
    """
    Dibuja las anotaciones de las líneas polinómicas sobre la imagen.
    :param image_path: Ruta de la imagen original.
    :param annotations_path: Ruta del archivo con las anotaciones (JSON).
    :param output_image_path: Ruta donde guardar la imagen con las anotaciones dibujadas.
    :param grid_size: Tamaño de la cuadrícula (por defecto 7).
    """
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Cargar las anotaciones
    with open(annotations_path, 'r') as f:
        annotations_data = json.load(f)
    
    if original_annotations_path is not None:
        with open(original_annotations_path, 'r') as f:
            original_annotations_data = json.load(f)
        
        image = draw_original_lines(image.copy(), original_annotations_data["Lines"])
    
    # Obtener la matriz de etiquetas (coeficientes de las líneas)
    label_matrix = np.array(annotations_data["label"])
    i = 0
    # Iterar sobre cada celda de la cuadrícula
    for grid_y in range(label_matrix.shape[0]):
        for grid_x in range(label_matrix.shape[1]):
            for b in range(label_matrix.shape[2] // 7):  # Asumiendo B = 2 y cada línea tiene 3 coeficientes
                param = label_matrix[grid_y, grid_x, b * 7: (b + 1) * 7]
                if np.any(param != 0):  # Solo dibujar si los coeficientes son diferentes de 0
                    # y_min = grid_y * (image.shape[0] // grid_size)
                    # y_max = (grid_y + 1) * (image.shape[0] // grid_size)
                    i += 1
                    print(i)
                    print(param)
                    image = draw_polynomial_line(image, param)
    
    # Guardar la imagen resultante
    cv2.imwrite(output_image_path, image)
    print(f"Imagen con anotaciones guardada en {output_image_path}")

name = "0a0d988cfa3c35dc6a7f90135c591148"
# name = "0a3b63d74afcd70349262ecc64d7a501"
image_path = f"/Users/danielserranodominguez/Desktop/Segmentor/Curvelanes/valid/images/{name}.jpg"
annotations_path = f"/Users/danielserranodominguez/Desktop/Segmentor/Curvelanes/valid/converted_labels_mal/{name}.lines.json"
original_annotations_path = f"/Users/danielserranodominguez/Desktop/Segmentor/Curvelanes/valid/labels/{name}.lines.json"
output_image_path = f"/Users/danielserranodominguez/Desktop/Segmentor/outputs/output_image_with_annotations_{name}.jpg"

plot_annotations_on_image(image_path, annotations_path, output_image_path, original_annotations_path)