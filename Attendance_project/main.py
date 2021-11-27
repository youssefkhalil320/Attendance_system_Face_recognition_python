import cv2
import numpy as np 
import face_recognition
import os 
from datetime import datetime

path = 'Attendance_images'
images = []
class_names = []
my_list = os.listdir(path)

for cl in my_list:
    curimg =cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    class_names.append(os.path.splitext(cl)[0])

def Encoder(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encode_list_known = Encoder(images)
print(len(encode_list_known))   

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            datestring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestring}')

cap = cv2.VideoCapture(1)
while True:
    _,img = cap.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    face_current_frame = face_recognition.face_locations(imgs)
    encodes_curr_frame = face_recognition.face_encodings(imgs,face_current_frame)

    for encode_face, face_locin in  zip(encodes_curr_frame, face_current_frame):
        matches = face_recognition.compare_faces(encode_list_known,encode_face)
        face_distance = face_recognition.face_distance(encode_list_known,encode_face)
        print(face_distance)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = class_names[match_index].upper()
            print(name)
            y1,x2,y2,x1 = face_locin
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)  
            markAttendance(name)      

    cv2.imshow('ori_image',img)
    cv2.waitKey(1)