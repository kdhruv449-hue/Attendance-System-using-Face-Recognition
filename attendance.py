import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# 1. Path to your images
path = 'Training_Images'
images = []
classNames = []

# Ensure the folder exists
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Created folder: {path}. Please add images and restart.")

myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

print(f"Found these people in database: {classNames}")

# 2. Advanced Marking Function (Checks Name AND Date)
def mark_attendance(name):
    # Using 'a+' allows reading and appending
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        my_data_list = f.readlines()
        
        now = datetime.now()
        date_string = now.strftime('%d-%m-%Y')
        time_string = now.strftime('%H:%M:%S')

        # Create a list of "Name-Date" to check for today's duplicates
        attendance_records = []
        for line in my_data_list:
            entry = line.split(',')
            if len(entry) >= 3:
                # Combine name and date from the file (stripping any newlines)
                attendance_records.append(f"{entry[0]}-{entry[2].strip()}")

        # Only log if this specific Name-Date combo doesn't exist
        current_entry_check = f"{name}-{date_string}"
        
        if current_entry_check not in attendance_records:
            # Add a newline if the file isn't empty and doesn't end with one
            f.writelines(f'\n{name},{time_string},{date_string}')
            f.flush() # Forces the data to save to the CSV immediately
            print(f"✅ Attendance logged for {name} on {date_string}")
        else:
            # This prevents the "Yesterday's data" bug
            pass 

# 3. Encoding Logic
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding faces... please wait.")
encodeListKnown = findEncodings(images)
print("Encoding Complete!")

# 4. Webcam Loop
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Faster processing by resizing
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

            # Draw Graphics
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # 5. Visual Clock
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(img, dt_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Attendance System', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
