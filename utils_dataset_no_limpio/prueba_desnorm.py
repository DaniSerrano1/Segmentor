import os
import numpy as np
import cv2

def denormalize_polynomial(coeffs, x_min, x_max, dy, w, h):
    """
    Desnormaliza un polinomio y las coordenadas x_min, x_max y dy a las dimensiones originales de la imagen.
    
    Args:
        coeffs: Lista [a, b, c, d] con coeficientes del polinomio normalizado
        x_min: Límite izquierdo del segmento normalizado
        x_max: Límite derecho del segmento normalizado
        dy: La variación en y normalizada
        w: Ancho original de la imagen
        h: Alto original de la imagen

    Returns:
        tuple: (new_coeffs, new_x_min, new_x_max, new_dy)
    """
    # Desnormalización de las coordenadas x_min y x_max
    x_min_denorm = x_min * w
    x_max_denorm = x_max * w

    # Desnormalización de los coeficientes del polinomio
    a, b, c, d = coeffs
    a_denorm = a * (w ** 3)
    b_denorm = b * (w ** 2)
    c_denorm = c * w
    d_denorm = d * h  # La coordenada y también se desnormaliza con respecto a la altura de la imagen

    # Desnormalización de dy
    dy_denorm = dy * h  # La variación en y también se desnormaliza con respecto a la altura

    return [a_denorm, b_denorm, c_denorm, d_denorm], x_min_denorm, x_max_denorm, dy_denorm

def draw_annot_in_image(coeffs, x_min, x_max, dy, image):
    """
    Desnormaliza los coeficientes, x_min, x_max, dy y dibuja la curva ajustada en la imagen.
    
    Args:
        coeffs: Lista de coeficientes normalizados [a_norm, b_norm, c_norm, d_norm]
        x_min: Límite izquierdo del segmento normalizado
        x_max: Límite derecho del segmento normalizado
        dy: Variación normalizada en y
        image: Imagen donde se dibujará la curva ajustada

    Returns:
        image: Imagen con la curva ajustada dibujada
    """
    # Desnormalizar los coeficientes, x_min, x_max, y dy
    [a_denorm, b_denorm, c_denorm, d_denorm], x_min_denorm, x_max_denorm, dy_denorm = denormalize_polynomial(coeffs, x_min, x_max, dy, image.shape[1], image.shape[0])

    # Dibujar la curva ajustada
    x_pred = np.linspace(x_min_denorm, x_max_denorm, 100).reshape(-1, 1)
    y_pred = (a_denorm * x_pred**3 + b_denorm * x_pred**2 + c_denorm * x_pred + d_denorm).flatten()  # Ecuación del polinomio

    x_pred = x_pred.flatten().astype(int)
    y_pred = y_pred.astype(int)

    # Dibujar los puntos y la curva ajustada en la imagen
    for i in range(len(x_pred) - 1):
        pt1, pt2 = (x_pred[i], y_pred[i]), (x_pred[i + 1], y_pred[i + 1])
        cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # Línea roja para la curva ajustada

    return image

def process_annotations_and_images(annotation_dir, image_dir, output_image_dir, resize_dim=(640, 640)):
    """
    Procesa las imágenes y sus anotaciones, redimensiona las imágenes, desnormaliza las anotaciones y dibuja las curvas.

    Args:
        annotation_dir: Directorio con los archivos .txt de las anotaciones normalizadas
        image_dir: Directorio con las imágenes
        output_image_dir: Directorio donde se guardarán las imágenes con las curvas dibujadas
        resize_dim: Tupla con las dimensiones (ancho, alto) a las que redimensionar las imágenes
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_image_dir, exist_ok=True)

    # Obtener los nombres de los archivos de anotaciones
    annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]

    for annotation_file in annotation_files:
        # Leer las anotaciones desde el archivo .txt
        annot_path = os.path.join(annotation_dir, annotation_file)
        with open(annot_path, 'r') as f:
            lines = f.readlines()
        
        # Obtener las anotaciones del archivo
        for line in lines:
            # Esperamos que la línea tenga la forma: a b c d x_min x_max dy
            parts = line.strip().split()
            coeffs = list(map(float, parts[:4]))  # coeficientes del polinomio [a, b, c, d]
            x_min, x_max, dy = map(float, parts[4:])  # x_min, x_max y dy

            # Obtener el nombre de la imagen correspondiente (con el mismo nombre que la anotación)
            image_name = annotation_file.replace('.txt', '.jpg')  # O usa el formato de imagen correspondiente
            image_path = os.path.join(image_dir, image_name)

            # Leer y redimensionar la imagen
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            image_resized = cv2.resize(image, resize_dim)

            # Dibujar la curva ajustada en la imagen redimensionada
            image_with_curve = draw_annot_in_image(coeffs, x_min, x_max, dy, image_resized)

            # Guardar la imagen con la curva ajustada
            output_image_path = os.path.join(output_image_dir, image_name)
            cv2.imwrite(output_image_path, image_with_curve)
            print(f"Imagen guardada en: {output_image_path}")

if __name__ == "__main__":
    WORKSPACE_PATH = "/Users/danielserranodominguez/Segmentor"
    annotation_dir = os.path.join(WORKSPACE_PATH, "Curvelanes/output/annots")
    image_dir = os.path.join(WORKSPACE_PATH, "Curvelanes/valid/images")
    output_image_dir = os.path.join(WORKSPACE_PATH, "Curvelanes/output/images")

    process_annotations_and_images(annotation_dir, image_dir, output_image_dir)