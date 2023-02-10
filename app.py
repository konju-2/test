import os
import requests

if not os.path.exists('./bertModel/pytorch_model.bin'):
    print('モデルをダウンロードします。')
    res = requests.get('https://www.dropbox.com/s/h4r1hiwtyg8n4wg/pytorch_model.bin?dl=1')
    with open('./bertModel/pytorch_model.bin', 'wb') as saveFile:
        saveFile.write(res.content)
    print('モデルを保存しました')
    
    
from flask import Flask, request
import platform
import word2music

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def route():
    print(request.form['text'])
    music_name = word2music.getMusic(request.form['text'])
    return music_name


@app.after_request
def after_request(response):
    allowed_origins = ['http://127.0.0.1:5500', 'https://it-engineer-k.github.io']
    origin = request.headers.get('Origin')
    print(origin)
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Headers', 'Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


if __name__ == '__main__' and platform.system() == 'Windows':
    app.run(debug=True)