from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from processing import process_image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

@app.route("/", methods=["GET", "POST"])
def index():
    original_image = None
    edge_image = None
    crater_image = None

    if request.method == "POST":
        image = request.files.get("image")

        if image:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            filename = secure_filename(image.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(image_path)

            edge_path, crater_path = process_image(image_path)

            original_image = os.path.relpath(image_path, BASE_DIR).replace("\\", "/")
            edge_image = os.path.relpath(edge_path, BASE_DIR).replace("\\", "/")
            crater_image = os.path.relpath(crater_path, BASE_DIR).replace("\\", "/")

    return render_template(
        "index.html",
        original_image=original_image,
        edge_image=edge_image,
        crater_image=crater_image
    )

# 🔥 CHATBOT ROUTE
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message").lower()

    responses = {
        "what is planex ai": "Planex AI is a lunar surface analysis system using image processing techniques.",
        "how does crater detection work": "It uses edge detection and Hough Circle Transform to identify crater-like structures.",
        "why moon": "The Moon provides high-resolution data and is suitable for proof-of-concept development.",
        "future work": "Future work includes CNN-based deep learning and expansion to other planets.",
        "cnn": "CNN is a deep learning model that automatically learns crater features from images."
    }

    for key in responses:
        if key in user_message:
            return jsonify({"reply": responses[key]})

    return jsonify({"reply": "Please ask about Planex AI, crater detection, Moon focus, CNN, or future work."})


if __name__ == "__main__":
    app.run(debug=True)
