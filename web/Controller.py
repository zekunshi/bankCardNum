
import os
import datetime

from flask import Flask, render_template, request
from crnn.predict import predict

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/upload', methods=["GET", "POST"])

def upload():
    startTime = datetime.datetime.now()

    print(startTime)
    root=r'./demo/test_images'
    im_names = os.listdir(root)

    i = 0

    while i < len(im_names):
        os.remove(root + '//' + im_names[i])

        i += 1

    # im_names = os.listdir('./test_images')
    # num_names = os.listdir('./cardNum')
    # i=0
    #
    # while i<len(im_names):
    #     os.remove('./test_images'+'//' + im_names[i])
    #     os.remove('./cardNum' + '//' + num_names[i])
    #     i+=1

    result = ""
    if request.method == 'POST':
        file = request.files['file']  # 获取文件信息用 request.FILES.get
        print(file.filename)  # 这里的get('file') 相当于 name = file
        file_name = file.filename
        file.save(os.path.join(root, file_name))
        result =  predict()
       # print('aaa')
    # 结束时间
    endTime = datetime.datetime.now()
    # 用时
    print(endTime - startTime)
    return result


def getPictureFromNet():
    app.run('0.0.0.0', 5000, debug=True)