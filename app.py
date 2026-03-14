import os
import cv2
import numpy as np
import uuid
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ===============================
# FOLDER SETUP
# ===============================

UPLOAD_FOLDER = os.path.join("static", "outputs")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ===============================
# CLEAR OLD OUTPUT FILES
# ===============================

def clear_output_folder():
    try:
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except:
        pass


# ===============================
# MAIN ROUTE
# ===============================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        clear_output_folder()

        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index.html")

        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_id + "_" + filename
        )

        file.save(filepath)

        img = cv2.imread(filepath)

        if img is None:
            return render_template("index.html", error="Image could not be processed")

        original = img.copy()

        # ===============================
        # PREPROCESSING
        # ===============================

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

        equalized = cv2.equalizeHist(gray)

        edges = cv2.Canny(blur, 50, 150)

        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        morphology = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

        # ===============================
        # CRATER DETECTION
        # ===============================

        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=100
        )

        crater_count = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))
            crater_count = len(circles[0])

            for circle in circles[0, :]:
                cv2.circle(
                    original,
                    (circle[0], circle[1]),
                    circle[2],
                    (0, 0, 255),
                    2
                )

        # ===============================
        # CONFIDENCE CALCULATION
        # ===============================

        if crater_count == 0:
            confidence = 40
        else:
            confidence = min(95, 60 + crater_count * 2)

        # ===============================
        # SAVE OUTPUT IMAGES
        # ===============================

        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "original.jpg"), img)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "gray.jpg"), gray)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "blur.jpg"), blur)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "equalized.jpg"), equalized)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "edges.jpg"), edges)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "threshold.jpg"), threshold)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "morphology.jpg"), morphology)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "crater.jpg"), original)

        outputs = {
            "original": "static/outputs/original.jpg",
            "gray": "static/outputs/gray.jpg",
            "blur": "static/outputs/blur.jpg",
            "equalized": "static/outputs/equalized.jpg",
            "edges": "static/outputs/edges.jpg",
            "threshold": "static/outputs/threshold.jpg",
            "morphology": "static/outputs/morphology.jpg",
            "crater": "static/outputs/crater.jpg"
        }

        return render_template(
            "index.html",
            outputs=outputs,
            crater_count=crater_count,
            confidence=confidence
        )

    return render_template("index.html")


# ===============================
# RUN SERVER
# ===============================

if __name__ == "__main__":
    app.run()