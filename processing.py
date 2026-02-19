import cv2
import os
import time
import numpy as np

def process_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # --------------------
    # EDGE DETECTION
    # --------------------
    edges = cv2.Canny(blur, 50, 150)

    # --------------------
    # CRATER DETECTION (HOUGH CIRCLE)
    # --------------------
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    crater_marked = img.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for (x, y, r) in circles:
            cv2.circle(crater_marked, (x, y), r, (0, 255, 0), 2)
            cv2.circle(crater_marked, (x, y), 2, (0, 0, 255), 3)

    # --------------------
    # SAVE OUTPUTS
    # --------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "static", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    ts = int(time.time())
    edge_path = os.path.join(output_dir, f"edges_{ts}.png")
    crater_path = os.path.join(output_dir, f"craters_{ts}.png")

    cv2.imwrite(edge_path, edges)
    cv2.imwrite(crater_path, crater_marked)

    return edge_path, crater_path
