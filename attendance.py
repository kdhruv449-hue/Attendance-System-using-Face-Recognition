import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# ==============================
# 1. PROJECT PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

training_path = os.path.join(BASE_DIR, "Training_Images")
attendance_file = os.path.join(BASE_DIR, "Attendance.csv")


# Create Training_Images folder if it doesn't exist
if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("Created Training_Images folder.")
    print("Please add face images and restart the program.")
    exit()


# ==============================
# 2. LOAD TRAINING IMAGES
# ==============================

images = []
classNames = []

myList = os.listdir(training_path)

for filename in myList:

    file_path = os.path.join(training_path, filename)

    # Read image
    curImg = cv2.imread(file_path)

    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(filename)[0])


print(f"Found these people in database: {classNames}")


# ==============================
# 3. MARK ATTENDANCE
# ==============================

def mark_attendance(name):

    with open(attendance_file, "a+", newline="") as f:

        f.seek(0)
        my_data_list = f.readlines()

        now = datetime.now()

        date_string = now.strftime("%d-%m-%Y")
        time_string = now.strftime("%H:%M:%S")

        attendance_records = []

        for line in my_data_list:

            entry = line.strip().split(",")

            if len(entry) >= 3:

                attendance_records.append(
                    f"{entry[0]}-{entry[2]}"
                )

        current_entry_check = f"{name}-{date_string}"

        if current_entry_check not in attendance_records:

            f.write(
                f"{name},{time_string},{date_string}\n"
            )

            print(
                f"✅ Attendance logged for {name} on {date_string}"
            )


# ==============================
# 4. FACE ENCODING
# ==============================

def findEncodings(images):

    encodeList = []

    for img in images:

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:

            encodeList.append(encodings[0])

        else:

            print(
                "⚠️ No face found in one of the training images."
            )

    return encodeList


print("Encoding faces... please wait.")

encodeListKnown = findEncodings(images)


if len(encodeListKnown) == 0:

    print("❌ No valid face encodings found.")
    print(
        "Please add clear face images "
        "to the Training_Images folder."
    )

    exit()


print("Encoding Complete!")


# ==============================
# 5. START WEBCAM
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Could not access the webcam.")

    exit()


while True:

    success, img = cap.read()

    if not success:
        print("❌ Could not read from webcam.")
        break


    # Resize image for faster processing

    imgS = cv2.resize(
        img,
        (0, 0),
        None,
        0.25,
        0.25
    )

    imgS = cv2.cvtColor(
        imgS,
        cv2.COLOR_BGR2RGB
    )


    # Detect faces

    facesCurFrame = face_recognition.face_locations(imgS)

    encodesCurFrame = face_recognition.face_encodings(
        imgS,
        facesCurFrame
    )


    # ==============================
    # 6. RECOGNIZE FACES
    # ==============================

    for encodeFace, faceLoc in zip(
        encodesCurFrame,
        facesCurFrame
    ):

        matches = face_recognition.compare_faces(
            encodeListKnown,
            encodeFace
        )

        faceDis = face_recognition.face_distance(
            encodeListKnown,
            encodeFace
        )

        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:

            name = classNames[matchIndex].upper()

            mark_attendance(name)


            # Face coordinates

            y1, x2, y2, x1 = faceLoc

            y1 *= 4
            x2 *= 4
            y2 *= 4
            x1 *= 4


            # Draw rectangle

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Name background

            cv2.rectangle(
                img,
                (x1, y2 - 35),
                (x2, y2),
                (0, 255, 0),
                cv2.FILLED
            )


            # Display name

            cv2.putText(
                img,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2
            )


    # ==============================
    # 7. DISPLAY DATE & TIME
    # ==============================

    now = datetime.now()

    dt_string = now.strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    cv2.putText(
        img,
        dt_string,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # ==============================
    # 8. DISPLAY CAMERA
    # ==============================

    cv2.imshow(
        "Attendance System",
        img
    )


    # Press Q to quit

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# 9. CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()