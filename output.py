import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date
import pickle
import training
from flask import Flask, render_template, Response
import csv

app=Flask(__name__)

camera = cv2.VideoCapture(0)

with open('Face_Recognition_Attendance_System/pickle_files/encodings.pickle', 'rb') as f:
    encodeListKnown = pickle.load(f)


def markAttendance(name):
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    filename = f"{current_date}.csv"
    
    if os.path.isfile(filename):
        with open(filename,'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            today = date.today()
            d1 = today.strftime("%d/%m/%Y")
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M')
                with open(filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, dtString, d1])
    else:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Time", "Date"])
            now = datetime.now()
            dtString = now.strftime('%H:%M')
            today = date.today()
            d1 = today.strftime("%d/%m/%Y")
            writer.writerow([name, dtString, d1])


def gen_frames():  
    while True:
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = training.classNames[matchIndex].upper()
        # print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)

            cv2.imshow('Webcam', img)
            cv2.waitKey(1)
            

            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)