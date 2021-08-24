import os
import io
import re
import nltk
import time
import h5py
import s3fs
import json
import boto3
import pickle
import pathlib
import requests
import tempfile


nltk.download('words')
nltk.download('stopwords')
from nltk.corpus import stopwords



import numpy as np
from flask import (Blueprint,
    render_template,
    Flask,request,
    session,
    redirect,
    url_for,
    
    send_file,abort)

from io import BytesIO 
from tensorflow import keras
from google.cloud import storage
from flask_session import Session
from keras.models import model_from_json
from werkzeug.utils import secure_filename
from keras.preprocessing.text import Tokenizer
from werkzeug.datastructures import  FileStorage
from keras.preprocessing.sequence import pad_sequences



words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

stop_words.remove('how')
stop_words.remove('where')
stop_words.remove('when')
stop_words.remove('who')
stop_words.remove('what')
stop_words.remove('which')
stop_words.remove('whom')


app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
Session(app)

S3_BUCKET     = "uploadedfilesfromtestui"
S3_KEY        = ""
S3_SECRET     = ""
S3_LOCATION   = "s3://uploadedfilesfromtestui/"




UPLOAD_URL = ""
CLOUD_STORAGE_BUCKET = ""


ALLOWED_EXTENSIONS = set(['wav','mp4','webm'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


def clean(text):
    text = text.lower()
    text = re.sub("mhm","",text)
    text = re.sub("hmm","",text)
    text = re.sub("yeah","",text)
    text = re.sub("'m"," am",text)
    text = re.sub("'s"," is",text)
    text = re.sub("'ll"," will",text)
    text = re.sub("'ve"," have",text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(w for w in nltk.wordpunct_tokenize(text)
         if w in words )
    text = " ".join(w for w in nltk.wordpunct_tokenize(text)
         if w not in stop_words)
    text = ' '.join(text.split())
    return text




@app.route('/upload', methods=['POST'])
def upload():

    uploaded_file = request.files.get('uploaded-file')
    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)


    uploaded_file.save(filename)
    

    bucket  = "uploadedfilesfromtestui"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET
    )

    try:
        s3.upload_file(Filename=filename,Bucket=bucket,Key=filename)
    except ClientError as e:
        logging.error(e)

    transcribe = boto3.client('transcribe',aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET)
    job_name = filename.replace(".mp4","")+"transcription_job"
    job_uri = S3_LOCATION + filename
    transcribe.start_transcription_job(
    TranscriptionJobName=job_name,
    Media={'MediaFileUri': job_uri},
    MediaFormat='mp4',
    LanguageCode='en-US',
    OutputBucketName= "uploadedfilesfromtestui"
    )

    
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
    time.sleep(5)
    pred=prediction(job_name)

    dict_ = {0:"How-To",1:"Questionnaire",2:"Review",3:"Event"}

    max_pred = np.argmax(pred,axis=0)

    return dict_[max_pred_index]



def prediction(job_name):

    
    s3 = boto3.resource(service_name='s3',aws_access_key_id=S3_KEY,
        aws_secret_access_key=S3_SECRET)
    obj = s3.Bucket('uploadedfilesfromtestui').Object(job_name+".json").get()
    temp = json.loads(obj['Body'].read().decode('utf-8'))
    text = temp['results']['transcripts'][0]['transcript']


    s3f = s3fs.S3FileSystem(anon=False, key=S3_KEY, secret=S3_SECRET)
    model_file = h5py.File(s3f.open("s3://testmodelfornlp/model2.h5", "rb"))
    loaded_model = keras.models.load_model(model_file)
    with BytesIO() as data:
        s3.Bucket("testmodelfornlp").download_fileobj("tokenizer.pickle", data)
        data.seek(0)    # move back to the beginning after writing
        tokenizer = pickle.load(data)

    ex_seq = tokenizer.texts_to_sequences(clean(text))
    ex_s_p = pad_sequences(ex_seq, maxlen=450)
    pred = loaded_model.predict(ex_s_p)

    return pred


    
if __name__ == '__main__':
    #app.run(ssl_context="adhoc")
    app.run(host='127.0.0.1', port=8080, debug=True)


