import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

training_path = os.path.join(BASE_DIR, "Training_Images")
attendance_file = os.path.join(BASE_DIR, "Attendance.csv")


# Create Training_Images folder if it doesn't exist
if not os.path.exists(training_path):
    os.makedirs(training_path)


# =========================
# LOAD TRAINING IMAGES
# =========================

image_extensions = (".jpg", ".jpeg", ".png")

myList = [
    file for file in os.listdir(training_path)
    if file.lower().endswith(image_extensions)
]

print("Found these people in database:", myList)


images = []
classNames = []

for cl in myList:

    image_path = os.path.join(training_path, cl)

    curImg = cv2.imread(image_path)

    if curImg is None:
        print(f"⚠️ Could not read image: {cl}")
        continue

    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


print("Encoding started... please wait.")


# =========================
# FIND FACE ENCODINGS
# =========================

def findEncodings(images, classNames):

    encodeList = []
    validNames = []

    for img, name in zip(images, classNames):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(imgRGB)

        if len(encodings) > 0:

            encodeList.append(encodings[0])
            validNames.append(name)

        else:

            print(f"⚠️ No face found in training image: {name}")

    return encodeList, validNames


encodeListKnown, classNames = findEncodings(images, classNames)

print("Encoding Complete!")


# =========================
# CHECK TRAINING IMAGES
# =========================

if len(encodeListKnown) == 0:

    print("\n❌ No valid face images found.")
    print("Please add clear face images inside the Training_Images folder.")
    print("Example: Training_Images/Dhruv.jpg")

    exit()


# =========================
# ATTENDANCE FUNCTION
# =========================

def markAttendance(name):

    today = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")

    # Create CSV if it doesn't exist
    if not os.path.exists(attendance_file):

        with open(attendance_file, "w") as f:
            f.write("Name,Date,Time\n")

    # Check whether person already marked attendance today
    with open(attendance_file, "r") as f:

        dataList = f.readlines()

        for line in dataList:

            if line.strip():

                existing_data = line.strip().split(",")

                if (
                    len(existing_data) >= 2
                    and existing_data[0] == name.upper()
                    and existing_data[1] == today
                ):
                    return

    # Add new attendance
    with open(attendance_file, "a") as f:

        f.write(f"{name.upper()},{today},{current_time}\n")

    print(f"✅ Attendance logged for {name.upper()} on {today}")


# =========================
# OPEN WEBCAM
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Could not open webcam.")
    print("Please check that your camera is connected and available.")

    exit()


print("\n📷 Camera started.")
print("Press 'Q' to quit.")


# =========================
# FACE RECOGNITION LOOP
# =========================

while True:

    success, img = cap.read()

    if not success:

        print("❌ Could not read frame from webcam.")
        break


    # Resize frame for faster processing
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    # Find faces
    facesCurFrame = face_recognition.face_locations(imgS)

    encodesCurFrame = face_recognition.face_encodings(
        imgS,
        facesCurFrame
    )


    # Compare detected faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

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

            # Scale face location back to original size
            y1, x2, y2, x1 = faceLoc

            y1 *= 4
            x2 *= 4
            y2 *= 4
            x1 *= 4


            # Draw rectangle around face
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Draw name background
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


            # Mark attendance
            markAttendance(name)


    # Display camera
    cv2.imshow("Face Attendance System", img)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

print("Camera closed.")
print("Attendance system stopped.")