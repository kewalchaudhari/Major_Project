# Dependecies
from __future__ import division, print_function
import os
import argparse
import numpy as np

# Image
import cv2
from sklearn.preprocessing import normalize
from PIL import Image, ImageStat

# Keras
import keras
import tensorflow as tf
from keras.layers import *
from keras.optimizers import SGD
from keras.models import load_model, Model
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.utils import img_to_array
from keras.models import load_model

from keras import backend as K
import datetime
import time

# Flask utils
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template

from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

# Config

# os.chdir("deploy/")
# Define a flask app
app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = os.urandom(12)

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'mysql'
app.config['MYSQL_DB'] = 'dred'

# Intialize MySQL
mysql = MySQL(app)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-w1", "--width", help="Target Image Width", type=int, default=256)
parser.add_argument("-h1", "--height",
                    help="Target Image Height", type=int, default=256)
parser.add_argument("-c1", "--channel",
                    help="Target Image Channel", type=int, default=1)
parser.add_argument("-p", "--path", help="Best Model Location Path",
                    type=str, default="D:\Engg Projects\FinalProject\Code\models\proj_model.h5")
parser.add_argument(
    "-s", "--save", help="Save Uploaded Image", type=bool, default=False)
parser.add_argument("--port", help="WSGIServer Port ID",
                    type=int, default=5000)
args = parser.parse_args()

SHAPE = (args.width, args.height, args.channel)
MODEL_SAVE_PATH = args.path
SAVE_LOADED_IMAGES = args.save

disease_copy = ''
patient_username = ''


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+K.epsilon()))

# SE block


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def create_model():

    dropRate = 0.3

    init = Input(SHAPE)
    x = Conv2D(32, (3, 3), activation=None, padding='same')(init)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation=None, padding='same')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(64, (5, 5), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(64, (3, 3), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x2 = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation=None, padding='same')(x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(128, (2, 2), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x3 = MaxPooling2D((2, 2))(x)

    ginp1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1)
    ginp2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(x2)
    ginp3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(x3)

    concat = Concatenate()([ginp1, ginp2, ginp3])
    gap = GlobalAveragePooling2D()(concat)

    x = Dense(256, activation=None)(gap)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropRate)(x)

    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(4, activation='softmax')(x)

    model = Model(init, x)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(
        lr=1e-3, momentum=0.9), metrics=['acc', precision, recall, f1])
    return model


model = create_model()
print(model.summary())
model.load_weights(MODEL_SAVE_PATH)
model_cg = load_model("D:\Engg Projects\FinalProject\Code\models\saved_model.model")
model_cg.make_predict_function()
graph = tf.compat.v1.get_default_graph()
print('Model loaded. Check http://localhost:{}/'.format(args.port))


def model_predict(img_path, model):
    global graph
    with graph.as_default():
        img = np.array(cv2.imread(img_path))
        img = cv2.resize(img, SHAPE[:2])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = normalize(img)
        img = np.expand_dims(img, axis=2)
        prediction = model.make_predict_function()
        prediction = model.predict(np.expand_dims(img, axis=0), batch_size=1)
        return prediction

# Threshold predictions


def threshold_arr(array):
    new_arr = []
    for ix, val in enumerate(array):
        loc = np.array(val).argmax(axis=0)
        k = list(np.zeros((len(val)), dtype=np.float16))
        k[loc] = 1
        new_arr.append(k)

    return np.array(new_arr, dtype=np.float16)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global disease_copy

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        oct_path = file_path

        im = Image.open(file_path).convert("RGB")
        stat = ImageStat.Stat(im)
        if sum(stat.sum)/3 != stat.sum[0]:
            orig = cv2.imread(oct_path)
            image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (64, 64))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            pred = model_cg.predict(image)
            pred = pred.argmax(axis=1)[0]

            if pred == 0:
                print("Cataract")
                result = "Cataract"
            elif pred == 1:
                print("Glaucoma")
                result = "Glaucoma"
            else:
                result = "Normal"
        else:
            result = "Normal"

        # ........................................................... .........
        disease_copy = result
        if not SAVE_LOADED_IMAGES:
            os.remove(file_path)

        return ""
    return None


@app.route('/')
def home():
    return render_template('video.html')  # return a string


@app.route('/options')
def welcome():
    return render_template('options.html')  # render a template

# after login should be after doctor login


@app.route('/AddP')
def AddPatient():
    return render_template('AddPatient.html')


@app.route('/index', methods=['post'])
def index():
    # Main page
    return render_template('AfterLogin.html')



@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('delete from accounts where id = 1;')
    return render_template('options.html')



@app.route('/result', methods=['POST'])
def result():
    global disease_copy
    print(disease_copy)
    if request.method == 'POST':
        time.sleep(5)
        name = request.form.get('name')
        age = request.form.get('age')
        mno = request.form.get('mno')
        addr = request.form.get('addr')
        today = str(datetime.date.today())
        iop = int(request.form.get('iop'))
        # print(iop)
        iop_result = "yes"
        if ((disease_copy == "Normal") and (iop <= 21)):
            iop_result = "Normal Eye"
        elif (disease_copy == "Normal" and (iop > 21 and iop < 70)):
            iop_result = "Possibility of Glaucomic Condition"
        elif (disease_copy == "Glaucoma" and (iop > 21 and iop < 70)):
            iop_result = "Glaucomic condition development"
        elif (disease_copy == "Glaucoma" and (iop >= 70)):
            iop_result = "You have glaucomic Condition"

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM patient_details WHERE pname = "'+name+'";')
        row = cursor.fetchone()
        if row:
            cursor.execute('update patient_details set diagnosis="' +
                           disease_copy+'" ,dov="'+today+'" where pname = "'+name+'";')
        else:
            cursor.execute('INSERT INTO patient_details VALUES (%s, %s, %s, %s, %s, %s, %s,%s)',
                           (name, age, mno, addr, today, disease_copy, iop,iop_result))
        mysql.connection.commit()
    return render_template('result.html', disease=disease_copy, name=name, Age=age, ph_no=mno, addr=addr, dov=today, iop=iop_result)

@app.route('/aboutus')
def about():
    return render_template('AboutUs.html')

@app.route('/success')
def success():
    l = []
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        'select * from patient_details where pname = "'+session['username']+'";')
    row = cursor.fetchone()

    if row:
        name1 = row[cursor.description[0][0]]
        age1 = row[cursor.description[1][0]]
        mno1 = row[cursor.description[2][0]]
        addr1 = row[cursor.description[3][0]]
        d = row[cursor.description[5][0]]
        dov1 = row[cursor.description[4][0]]
        
        
        
        return render_template('user_result.html', disease=d, name=name1, Age=age1, ph_no=mno1, addr=addr1, dov=dov1)
    else:
        return render_template("no_records_found.html")


# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/LoginUser/', methods=['GET', 'POST'])
def userlogin():
    # Output message if something goes wrong...
    global patient_username
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM user_login WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # return render_template('result.html')
            return redirect(url_for('success'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('user_login.html', msg=msg)


@app.route('/LoginDoctor', methods=['GET', 'POST'])
def doctorlogin():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
    # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM doctor_login WHERE username = %s AND password = %s', (username, password))
    # Fetch one record and return result
        account = cursor.fetchone()
    # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['pid'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            # return 'Logged in successfully!'
            return render_template('AfterLogin.html')
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('doctor_login.html', msg=msg)


@app.route('/RegUser', methods=['GET', 'POST'])
def user_register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        pname = request.form['pname']
        page = request.form['page']
        pcontact = request.form['p_contact']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM user_login WHERE username LIKE %s', [username])
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO user_login VALUES (NULL, %s, %s, %s, %s, %s)',
                           (username, password, pname, page, pcontact))
            mysql.connection.commit()
            msg = 'You have successfully registered!Clik on login button to proceed'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('user_register.html', msg=msg)


@app.route('/RegDoc', methods=['GET', 'POST'])
def doctor_register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        dname = request.form['dname']
        dcontact = request.form['dcontact']
        specialization = request.form['specialization']
        email = request.form['email']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM doctor_login WHERE username LIKE %s', [username])
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO doctor_login VALUES (NULL, %s, %s, %s, %s, %s, %s)',
                           (username, password, dname, specialization, dcontact, email))
            mysql.connection.commit()
            msg = 'You have successfully registered! Click on login button to proceed'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
        # Show registration form with message (if any)
    return render_template('doctor_register.html', msg=msg)


if __name__ == '__main__':
    app.run(debug=True)
    http_server = WSGIServer(('', args.port), app)
    http_server.serve_forever()
