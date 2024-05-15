#Flask入門-1.2 最小規模のFlaskアプリを実行してみよう ※たぶん
# from flask import Flask, render_template
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run()

#Flask入門-1.3 MNISTを用いた数字認識アプリ制作
import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np


classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# model = load_model('./model.h5')#学習済みモデルをロード
model = load_model('/Users/misz8/Aidemy_ONGAESHI/mnist_app/model.h5')#学習済みモデルをロード

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            # img = image.load_img(filepath, grayscale=True, target_size=(image_size,image_size)) #「grayscale」は引数として使えなくなった
            img = image.load_img(filepath, color_mode="grayscale", target_size=(image_size,image_size))            
            img = image.img_to_array(img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    #Flask入門-1.3 MNISTを用いた数字認識アプリ制作
    # app.run()
    #Flask入門-1.4 公開設定
    port = int(os.environ.get('PORT', 8080))
    app.run(host = '0.0.0.0', port = port)
