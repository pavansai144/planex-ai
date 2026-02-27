import cv2
import os
import time

def process_image(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours_img = img.copy()
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 1)

    crater_img = img.copy()
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    crater_count = 0
    confidence = 0

    if circles is not None:
        circles = circles[0]
        crater_count = len(circles)

        for (x, y, r) in circles:
            cv2.circle(crater_img, (int(x), int(y)), int(r), (0, 0, 255), 2)

        confidence = min(95, 50 + crater_count * 5)
    else:
        confidence = 40

    timestamp = str(int(time.time()))
    outputs = {}

    def save(image, name):
        filename = f"{name}_{timestamp}.png"
        path = os.path.join("static/outputs", filename)
        cv2.imwrite(path, image)
        return path

    outputs["gray"] = save(gray, "gray")
    outputs["blur"] = save(blur, "blur")
    outputs["edges"] = save(edges, "edges")
    outputs["thresh"] = save(thresh, "threshold")
    outputs["contours"] = save(contours_img, "contours")
    outputs["crater"] = save(crater_img, "crater")

    return outputs, crater_count, confidence