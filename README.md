# Face Attendance System

An automated attendance system built with Python, OpenCV, NumPy, and face recognition technology. The system identifies registered faces through a webcam and automatically records attendance with the person's name, date, and time.

## Features

- Real-time face detection and recognition
- Automatic attendance marking
- Prevents duplicate attendance on the same day
- Records name, date, and time in a CSV file
- Supports multiple registered people
- Automatically creates the attendance file
- Works with a local webcam
- Easy to set up and use

## Technologies Used

- Python
- OpenCV
- NumPy
- Face Recognition
- dlib
- CSV

## Project Structure

```text
Face_Attendance_System/
│
├── Training_Images/
│   └── README.md
│
├── attendance.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. The system loads face images from the `Training_Images` folder.
2. It generates face encodings for the registered people.
3. The webcam captures video in real time.
4. The system detects faces from the webcam.
5. Detected faces are compared with the registered face encodings.
6. When a registered person is recognized, their attendance is automatically recorded.
7. The attendance is saved with the person's name, date, and time.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kdhruv449-hue/Attendance-System-using-Face-Recognition.git
```

### 2. Open the Project Folder

```bash
cd Attendance-System-using-Face-Recognition
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Add Face Images

Open the `Training_Images` folder and add clear face images of the people you want to register.

For example:

```text
Training_Images/
├── Aman.jpg
├── Dhruv.jpg
└── Priya.jpg
```

The **filename is used as the person's name**.

For example:

```text
Dhruv.jpg → DHRUV
Aman.jpg  → AMAN
Priya.jpg → PRIYA
```

### Image Guidelines

- Use one person per image.
- Use a clear and front-facing photo.
- Make sure the face is clearly visible.
- Supported formats include `.jpg`, `.jpeg`, and `.png`.
- Do not use images without a detectable face.

## Run the Project

Run:

```bash
python attendance.py
```

The webcam will open automatically.

When a registered face is recognized, attendance will be recorded.

Press:

```text
Q
```

to close the webcam.

## Attendance Records

The system automatically creates:

```text
Attendance.csv
```

The file contains:

```text
Name, Date, Time
```

Example:

```text
DHRUV, 17-09-2026, 15:30:21
AMAN, 17-09-2026, 15:32:10
```

The `.gitignore` file prevents attendance records from being uploaded to GitHub.

## Requirements

The project requires:

- Python 3
- Webcam
- OpenCV
- NumPy
- face-recognition
- dlib

Install all dependencies using:

```bash
python -m pip install -r requirements.txt
```

## Important Notes

- A working webcam is required.
- Face images should be clear and properly visible.
- The project is designed to run locally on a computer.
- Do not upload personal face images or attendance records to a public repository without appropriate permission.

## Future Improvements

- Web-based interface
- Database integration
- Admin dashboard
- Multiple camera support
- Cloud-based attendance storage
- Improved face recognition performance
- Export attendance reports
- User authentication

## Usage / Disclaimer

This project is intended for educational and demonstration purposes. Use face recognition and attendance data responsibly and obtain appropriate consent when using personal biometric information.

## Author

**Dhruv Kumar**

B.Tech – Electronics and Communication Engineering  
Dr. B. R. Ambedkar National Institute of Technology, Jalandhar
