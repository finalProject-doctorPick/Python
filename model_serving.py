import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# 모델 로드
model_dir = './pb'
loaded_model = tf.saved_model.load(model_dir)


def preprocess_image(image_path, target_size=(416, 416)):
    img = Image.open(image_path)
    original_size = img.size

    # 이미지 리사이징
    img.thumbnail(target_size, Image.ANTIALIAS)
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2))

    img = np.array(background, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img, original_size

def parse_prediction(predictions):
    labels = ['drugs']
    objects = []
    for i in range(predictions.shape[1]):
        label = labels[int(predictions[0, i, 4])]
        confidence = float(predictions[0, i, 4])
        bbox = predictions[0, i, :4].numpy().tolist()
        objects.append({
            'label': label,
            'confidence': confidence,
            'bbox': bbox
        })
    return {'objects': objects}

def draw_boxes(image_path, objects, original_size, original_filename):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for obj in objects:
        bbox = obj['bbox']
        label = obj['label']
        confidence = round(obj['confidence'], 2)

        # Scale bbox coordinates to match original image size
        bbox = [bbox[0] * original_size[0], bbox[1] * original_size[1], bbox[2] * original_size[0], bbox[3] * original_size[1]]
        draw.rectangle(bbox, outline='red')

        # Add label and confidence text above the bounding box
        text = f"{label}: {confidence}"
        draw.text((bbox[0], bbox[1] - 15), text, fill='red')

    result_path = f'static/{original_filename}'
    img.save(result_path)
    return result_path

@app.route('/')
def index():
    return render_template('yolo_search.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    _, temp_filename = tempfile.mkstemp(suffix=".jpg")
    image.save(temp_filename)

    input_image, original_size = preprocess_image(temp_filename)
    original_filename = secure_filename(image.filename)
    predictions = loaded_model(input_image)
    result = parse_prediction(predictions)
    result_image_path = draw_boxes(temp_filename, result['objects'], original_size, original_filename)

    return render_template('yolo_result.html', image_url=f'/static/{original_filename}')

if __name__ == '__main__':
    app.run(debug=True)
