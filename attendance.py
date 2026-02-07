import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# ==============================
# 1. LOAD TRAINING IMAGES
# ==============================
path = 'Training_Images'
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print("Registered Faces:", classNames)

# ==============================
# 2. ATTENDANCE FUNCTION (CSV)
# ==============================
def mark_attendance(name):
    today_date = datetime.now().strftime('%d-%m-%Y')
    file_name = f'Attendance_{today_date}.csv'

    # Create file & header if not exists
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            f.write('Name,Time,Date\n')

    with open(file_name, 'r+') as f:
        lines = f.readlines()
        name_list = []

        for line in lines[1:]:
            entry = line.split(',')
            name_list.append(entry[0])

        if name not in name_list:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            f.write(f'{name},{time_string},{today_date}\n')
            print(f"Attendance marked for {name}")

# ==============================
# 3. ENCODING FUNCTION
# ==============================
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding faces...")
encodeListKnown = findEncodings(images)
print("Encoding complete!")

# ==============================
# 4. WEBCAM FACE RECOGNITION
# ==============================
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            mark_attendance(name)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "UNKNOWN", (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Show date & time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(img, dt_string, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Face Recognition Attendance System', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
