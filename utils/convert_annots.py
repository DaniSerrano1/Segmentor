import json
import os
import numpy as np
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm

def scale_points(points, original_resolution, target_resolution):
    """Escala los puntos de la resolución original a la resolución objetivo."""
    scale_x = target_resolution[0] / original_resolution[0]
    scale_y = target_resolution[1] / original_resolution[1]
    
    scaled_points = [(p[0] * scale_x, p[1] * scale_y) for p in points]
    return scaled_points

def fit_polynomial(points, degree, frame):
    """Ajusta un polinomio a los puntos y dibuja la curva en la imagen."""
    x = np.array([p[0] for p in points]).reshape(-1, 1)  
    y = np.array([p[1] for p in points]).reshape(-1, 1) 

    model = LinearRegression()
    
    if degree == 1:
        x_poly = x  
    else:
        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x)

    model.fit(x_poly, y)

    if degree == 1:
        c = model.coef_[0, 0]
        d = model.intercept_[0]
        a, b = 0, 0  
    else:
        a, b, c, d = model.coef_[0, -1], model.coef_[0, -2], model.coef_[0, -3], model.intercept_[0]

    x_min, x_max = int(min(x).item()), int(max(x).item())
    dy = int((max(y) - min(y)).item())

    x_pred = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_pred = model.predict(poly.transform(x_pred)) if degree > 1 else model.predict(x_pred)

    x_pred = x_pred.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)

    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), 5)

    for i in range(len(x_pred) - 1):
        pt1, pt2 = (x_pred[i], y_pred[i]), (x_pred[i + 1], y_pred[i + 1])
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    return a, b, c, d, x_min, x_max, dy, frame

def image_processing(image_path, annotation_path, output_txt_path, degree=3, target_resolution=(854, 480)):
    if not os.path.exists(image_path) or not os.path.exists(annotation_path):
        print(f"Error: Archivo no encontrado -> {image_path} o {annotation_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen -> {image_path}")
        return None
    
    original_resolution = (image.shape[1], image.shape[0])
    image = cv2.resize(image, target_resolution)
    
    with open(annotation_path, "r") as f:
        label_data = json.load(f)

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    with open(output_txt_path, "w") as f_out:
        for line in label_data.get("Lines", []):
            points = [(float(p["x"]), float(p["y"])) for p in line]
            scaled_points = scale_points(points, original_resolution, target_resolution)

            fit_degree = degree if len(line) > 2 else 1
            a, b, c, d, x_min, x_max, dy, image = fit_polynomial(scaled_points, fit_degree, image)

            f_out.write(f"{a} {b} {c} {d} {x_min} {x_max} {dy}\n")

    return image

if __name__ == "__main__":
    WORKSPACE_PATH = "/Users/danielserranodominguez/Segmentor"
    annotations_path = os.path.join(WORKSPACE_PATH, "Curvelanes/valid/labels")   
    images_path = os.path.join(WORKSPACE_PATH, "Curvelanes/valid/images")
    output_annotations_path = os.path.join(WORKSPACE_PATH, "Curvelanes/output/annots")
    output_images_path = os.path.join(WORKSPACE_PATH, "Curvelanes/output/out_images")

    os.makedirs(output_annotations_path, exist_ok=True)
    os.makedirs(output_images_path, exist_ok=True)

    for i, annot_name in tqdm(enumerate(os.listdir(annotations_path)), desc="Processing images", total=len(os.listdir(annotations_path))):
        annotation_path = os.path.join(annotations_path, annot_name)
        image_path = annotation_path.replace("labels", "images").replace(".lines.json", ".jpg")
        output_txt_path = os.path.join(output_annotations_path, annot_name.replace(".lines.json", ".txt"))
        output_img_path = os.path.join(output_images_path, os.path.basename(image_path))

        processed_image = image_processing(image_path, annotation_path, output_txt_path)
        
        # cv2.imwrite(output_img_path, processed_image)
        
        # if i == 5:
        #     break