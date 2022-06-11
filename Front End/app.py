from flask import Flask, render_template, request, url_for
import threading,time

import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('./model/model.h5', custom_objects=None, compile=True, options=None)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path ="./images/"+imagefile.filename
    imagefile.save(image_path)


    img = tf.keras.utils.load_img(image_path, target_size=(186, 186))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    class_pred = np.argmax(predictions,axis=1)


    for i in class_pred:
        if (i==2):
            prediction = 'tan(x)'
        elif (i==1):
            prediction = 'sin(x)'
        elif (i==0):
           prediction = 'cos(x)'


    return render_template('index.html', graph = prediction)


if __name__ == '__main__':
    app.run(debug=True)