import os
import csv
import base64
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
import face_recognition

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(BASE_DIR, "Training_Images")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "Attendance.csv")

known_encodings = []
known_names = []


def load_training_images():
    known_encodings.clear()
    known_names.clear()

    if not os.path.isdir(TRAINING_DIR):
        return

    for filename in os.listdir(TRAINING_DIR):
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        path = os.path.join(TRAINING_DIR, filename)

        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(
                    os.path.splitext(filename)[0].upper()
                )
        except Exception as exc:
            print(f"Could not process {filename}: {exc}")


def mark_attendance(name):
    today = datetime.now().strftime("%d-%m-%Y")
    now = datetime.now().strftime("%H:%M:%S")

    # Avoid duplicate attendance for the same day.
    if os.path.exists(ATTENDANCE_FILE):
        with open(
            ATTENDANCE_FILE, newline="", encoding="utf-8"
        ) as file:
            for row in csv.DictReader(file):
                if (
                    row.get("Name") == name
                    and row.get("Date") == today
                ):
                    return False

    file_exists = os.path.exists(ATTENDANCE_FILE)

    with open(
        ATTENDANCE_FILE, "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        writer.writerow([name, today, now])

    return True


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    if not known_encodings:
        return jsonify({
            "status": "error",
            "message": "No valid training faces are available."
        }), 503

    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")

    if not image_data.startswith("data:image/"):
        return jsonify({
            "status": "error",
            "message": "A valid camera image is required."
        }), 400

    try:
        encoded = image_data.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)

        if len(raw) > 4_000_000:
            return jsonify({
                "status": "error",
                "message": "Image is too large."
            }), 413

        image_array = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                "status": "error",
                "message": "Could not read the camera image."
            }), 400

        # Reduce image size for faster processing.
        small = cv2.resize(
            frame, (0, 0), fx=0.25, fy=0.25
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(
            rgb, locations
        )

        for encoding in encodings:
            distances = face_recognition.face_distance(
                known_encodings, encoding
            )
            best_index = int(np.argmin(distances))

            # Require a reasonably close match.
            if distances[best_index] < 0.50:
                name = known_names[best_index]
                added = mark_attendance(name)

                return jsonify({
                    "status": "recognized",
                    "name": name,
                    "attendance": (
                        "Attendance marked"
                        if added
                        else "Already marked today"
                    )
                })

        return jsonify({
            "status": "unknown",
            "message": "No registered face recognized."
        })

    except Exception as exc:
        app.logger.exception("Recognition failed")
        return jsonify({
            "status": "error",
            "message": "Could not process the image."
        }), 500


load_training_images()
