from flask import Flask, render_template, request, send_file
from ultralytics import YOLO  # Import for YOLOv8
import cv2
import torch
import os

app = Flask(__name__)

# Load the YOLOv8 model when the Flask app starts
MODEL_PATH = '/home/lloydhangoma/Desktop/API/best.pt'
model = YOLO(MODEL_PATH)

# Flask route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Flask route to handle image upload and prediction
@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

    # Make a prediction
    with torch.no_grad():
        results = model(image_rgb)

    # Access the first result (YOLOv8 returns a list of predictions)
    predictions = results[0]

    # Draw bounding boxes on the image
    for box in predictions.boxes.xyxy.cpu().numpy():
        x_min, y_min, x_max, y_max = map(int, box[:4])  # Convert to integer values
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Draw rectangle

    # Save the image with bounding boxes
    output_image_path = "./images/output_" + imagefile.filename
    cv2.imwrite(output_image_path, image)

    # Serve the image with bounding boxes
    return send_file(output_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs("./images")
    app.run(port=3001, debug=True)
