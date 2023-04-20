import cv2
import numpy as np
import face_recognition
import os
import pickle


path = 'Face_Recognition_Attendance_System\Training_images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(tuple(encode))  # convert numpy array to tuple
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

if os.path.exists('Face_Recognition_Attendance_System/pickle_files/encodings.pickle'):
    with open('Face_Recognition_Attendance_System/pickle_files/encodings.pickle', 'rb') as f:
        encodeListOld = pickle.load(f)
    if set(map(tuple, encodeListOld)) != set(map(tuple, encodeListKnown)):
        with open('Face_Recognition_Attendance_System/pickle_files/encodings.pickle', 'wb') as f:
            pickle.dump(encodeListKnown, f)
        print('Pickle file updated.')
    else:
        print('Pickle file is up-to-date.')
else:
    with open('Face_Recognition_Attendance_System/pickle_files/encodings.pickle', 'wb') as f:
        pickle.dump(encodeListKnown, f)
    print('Pickle file created.')