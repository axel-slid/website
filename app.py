
# %%
# app.py

import os
import numpy as np
import cv2
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
import albumentations as A
import segmentation_models as sm
import tensorflow as tf
from werkzeug.utils import secure_filename

os.environ['SM_FRAMEWORK'] = 'tf.keras'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PREDICTIONS_FOLDER'] = 'predictions'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTIONS_FOLDER'], exist_ok=True)

BACKBONE = 'inceptionv3'
n_classes = 1
activation = 'sigmoid'

model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model.load_weights('best_model.h5')

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or invalid image format: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_preprocessing(preprocessing_fn):
    return A.Compose([A.Lambda(image=preprocessing_fn)])

def predict_image(model, image_path, save_path):
    image = load_image(image_path)
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, (512, 512))
    preprocess_input = sm.get_preprocessing(BACKBONE)
    preprocessing = get_preprocessing(preprocess_input)
    sample = preprocessing(image=image_resized)
    image_prep = sample['image']
    image_prep = np.expand_dims(image_prep, axis=0)
    pr_mask = model.predict(image_prep).round().squeeze()
    pr_mask_resized = cv2.resize(pr_mask, (original_size[1], original_size[0]))
    pr_mask_resized_uint8 = (pr_mask_resized * 255).astype(np.uint8)
    if not os.path.splitext(save_path)[1]:
        save_path += '.png'
    cv2.imwrite(save_path, pr_mask_resized_uint8)
    white_pixels = np.sum(pr_mask_resized_uint8 == 255)
    total_pixels = pr_mask_resized_uint8.size
    percentage_microplastic = (white_pixels / total_pixels) * 100
    has_microplastic = white_pixels > 0
    return f"{percentage_microplastic:.2f}", has_microplastic

class UploadForm(FlaskForm):
    file = FileField('Image')
    submit = SubmitField('Upload')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segmenter', methods=['GET', 'POST'])
def segmenter():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        base_filename = os.path.splitext(filename)[0]
        prediction_filename = f'pred_{base_filename}.png'
        prediction_path = os.path.join(app.config['PREDICTIONS_FOLDER'], prediction_filename)
        percentage_microplastic, has_microplastic = predict_image(model, filepath, prediction_path)
        return redirect(url_for('display_image', filename=filename, percentage=percentage_microplastic, has_microplastic=has_microplastic))
    return render_template('segmenter.html', form=form)

@app.route('/display/<filename>')
def display_image(filename):
    original_image_url = url_for('uploaded_file', filename=filename)
    base_filename = os.path.splitext(filename)[0]
    prediction_filename = f'pred_{base_filename}.png'
    prediction_image_url = url_for('prediction_file', filename=prediction_filename)
    percentage = request.args.get('percentage')
    has_microplastic = request.args.get('has_microplastic')
    return render_template('display.html', original_image_url=original_image_url, prediction_image_url=prediction_image_url, percentage=percentage, has_microplastic=has_microplastic)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predictions/<filename>')
def prediction_file(filename):
    return send_from_directory(app.config['PREDICTIONS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

# %%
