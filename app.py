from flask import Flask,redirect,render_template,url_for

import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['post','get'])
def predict():
    cap=cv2.VideoCapture(-1)

    while True:
        ret,img=cap.read()
        if ret == False:
            break

        gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray)

        for(x,y,w,h)in faces:
            
            

            cv2.putText(img,'face',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(200,255,255),2,cv2.LINE_AA)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

            roi_gray=gray[y:y+h,x:x+w]
            roi_color=img[y:y+h,x:x+w]

            eyes=eye_cascade.detectMultiScale(roi_gray)

            for(ex,ey,ew,eh) in eyes:
                
                

                cv2.putText(roi_color,'eye',(ex,ey),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,255,255),2,cv2.LINE_AA)
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

        cv2.imshow('Face & Eye Detection',img)

        if cv2.waitKey(1) & 0xFF==ord('w'):
            break
    return render_template('index.html')
    cap.release() 
    cv2.destroyAllWindows() 
    
if __name__=='__main__':
    app.run(debug=True)
