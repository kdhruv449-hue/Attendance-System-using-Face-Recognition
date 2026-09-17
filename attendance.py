import cv2
import numpy as np
import face_recognition
import os

# 1. Path to your images
path = 'Training_Images'
images = []
classNames = []
myList = os.listdir(path)

# 2. Automatically get names from filenames
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(f"Found these people: {classNames}")

# 3. Create a function to 'encode' (memorize) the faces
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

print("Encoding started... please wait.")
encodeListKnown = findEncodings(images)


if len(encodeListKnown) == 0:

    print("❌ No valid face encodings found.")
    print(
        "Please add clear face images "
        "to the Training_Images folder."
    )

    exit()


print("Encoding Complete!")



# 4. Start the Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Could not access the webcam.")

    exit()


while True:

    success, img = cap.read()
    # Resize image to 1/4 size for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find faces in the current webcam frame
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
            
            # Scale face locations back up to original size
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            
            # Draw the box and the name
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Show the webcam window
    cv2.imshow('Webcam', img)
    
    # Press 'q' to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==============================
# 9. CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()
