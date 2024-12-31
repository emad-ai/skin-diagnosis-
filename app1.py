from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import keras.models
from tensorflow.keras.preprocessing import image
import numpy as np
import jinja2
import tensorflow as tf 

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# تحميل النموذج المدرب
model = tf.keras.models.load_model('C:/Users/pc/Desktop/121/resnet101CNN.keras') 
  
# تصنيف الصورة
def get_prediction(img, model, labels=[
    'Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox'
  ], target_size=(224, 224)):
  class_names = labels
  img = tf.keras.utils.load_img(img, target_size=target_size)
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  prediction = model.predict(img_array)
  score = [tf.nn.softmax(prediction)[0][i].numpy() * 100  for i in range(len(class_names))]
  result = sorted([(class_names[i], f"{score[i]:.2f}%") for i in range(len(class_names))], key=lambda x: float(x[1].replace('%','')), reverse=True)
  highest_label = result[0][0]  # الفئة الأعلى احتمالاً
  return result, highest_label

# حفظ الصورة في المجلد المناسب
def save_image_to_class_folder(file_path, label):
    class_folder = os.path.join(app.config['UPLOAD_FOLDER'], label)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)  # إنشاء المجلد إذا لم يكن موجودًا
    new_file_path = os.path.join(class_folder, os.path.basename(file_path))
    os.rename(file_path, new_file_path)  # نقل الصورة إلى المجلد المناسب
    return new_file_path

# الصفحة الرئيسية
@app.route('/')
def home():
    return render_template('index.html')

# صفحة التشخيص
@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # الحصول على التوقع والفئة
            result, highest_label = get_prediction(file_path, model)

            # حفظ الصورة في المجلد المناسب
            saved_file_path = save_image_to_class_folder(file_path, highest_label)

            return render_template('diagnose.html', diagnosis=result, image_url=saved_file_path)
    return render_template('diagnose.html', diagnosis=None)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)