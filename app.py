import os
from flask import Flask, request, redirect, url_for,flash,render_template
from werkzeug.utils import secure_filename
import time
import cv2
from get_train_embedding import *
predict = None
from process_Image import *
def createFolder(label):
    try:
       os.mkdir("mydata/" + label)
    except:
       print("folder created!!!!")
UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/test",methods=['GET','POST'])
def upload_file():
    global predict
    if request.method == "GET":
      return render_template("test.html")
    if request.method == 'POST':
        src=None
        # check if the post request has the file part
        if 'fileToUpload' not in request.files:
            return redirect(request.url)
        label = request.form.get('label', '')
        file = request.files['fileToUpload']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print("no selected file!!!!!!!!!!!!!!!!!!!!!")
            flash('No selected file')
            return redirect(request.url)
        if file.filename == '':
            print("no label !!!!!!!!!!!!!!!!!!!!!")
            flash('No label ')
            return redirect(request.url)
        if file:

            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"input.jpg"))
            src = "static/input.jpg"
            img = cv2.imread(src)
            predict.setImage(img)
            predict.recogniton()
         # cv2.imwrite("image",img)
            cv2.imwrite("static/res.jpg", img)
            img = predict.image
            # cv2.imwrite("image",img)
            cv2.imwrite("static/res.jpg",img) 
        return render_template("test.html", src="static/res.jpg")

@app.route("/train",methods=['GET','POST'])
def train():
    if request.method == "GET":
      return render_template("train.html")
    if request.method == 'POST':
        src=None
        # check if the post request has the file part
        if True:
            src = None
            print("A")
            # check if the post request has the file part
            if 'fileToUpload' not in request.files:
                return redirect(request.url)
            label = request.form.get('label', '')
            file = request.files['fileToUpload']
            # if user does not select file, browser also
            # submit a empty part without filename
        print(file)
        if file.filename == '':
            print("no selected file!!!!!!!!!!!!!!!!!!!!!")
            flash('No selected file')
            return redirect(request.url)
        if file.filename == '':
            print("no label !!!!!!!!!!!!!!!!!!!!!")
            flash('No label ')
            return redirect(request.url)
        if file and label:
            createFolder(label)
            path="mydata/"+label+"/"+str(time.time())+".jpg"
            file.save(os.path.join("",path))
            train.train()
        return render_template("train.html")  
if __name__ == "__main__":
    predict = Process()
    train=Traindata()
    app.run(debug=True,host="0.0.0.0",port=9000)
