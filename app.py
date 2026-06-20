import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import mediapipe as mp

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Smart Attendance Dashboard",
    layout="wide"
)

st.title("📊 Smart Attendance Dashboard + AI Camera")

menu = ["Dashboard", "AI Camera", "Analytics"]

choice = st.sidebar.selectbox("Menu", menu)

# ---------------- FILE ---------------- #

ATTENDANCE_FILE = "Attendance.csv"

# ---------------- LOAD IMAGES ---------------- #

path = 'ImagesAttendance'

images = []
classNames = []

myList = os.listdir(path)

for cl in myList:

    curImg = cv2.imread(f'{path}/{cl}')

    images.append(curImg)

    classNames.append(os.path.splitext(cl)[0])

# ---------------- FACE ENCODINGS ---------------- #

def findEncodings(images):

    encodeList = []

    for img in images:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        encode = face_recognition.face_encodings(img)[0]

        encodeList.append(encode)

    return encodeList

encodeListKnown = findEncodings(images)

print("Encoding Complete")

# ---------------- MESSAGE CONTROL ---------------- #

last_message = ""

# ---------------- ATTENDANCE FUNCTION ---------------- #

def markAttendance(name):

    global last_message

    now = datetime.now()

    current_date = now.strftime('%Y-%m-%d')

    current_time = now.strftime('%H:%M:%S')

    # Create file if not exists

    if not os.path.exists(ATTENDANCE_FILE):

        df = pd.DataFrame(
            columns=['Name', 'Date', 'Time']
        )

        df.to_csv(ATTENDANCE_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_FILE)

    # Check today's attendance

    already_marked = (
        (df['Name'] == name) &
        (df['Date'] == current_date)
    ).any()

    if not already_marked:

        new_row = pd.DataFrame(
            [[name, current_date, current_time]],
            columns=['Name', 'Date', 'Time']
        )

        df = pd.concat(
            [df, new_row],
            ignore_index=True
        )

        df.to_csv(
            ATTENDANCE_FILE,
            index=False
        )

        # Show success once

        if last_message != f"success_{name}":

            st.success(
                f"✅ Attendance marked for {name}"
            )

            last_message = f"success_{name}"

    else:

        # Show warning once

        if last_message != f"warning_{name}":

            st.warning(
                f"⚠️ {name} attendance already marked today"
            )

            last_message = f"warning_{name}"

# ---------------- DASHBOARD ---------------- #

if choice == "Dashboard":

    st.subheader("📋 Daily Attendance")

    if os.path.exists(ATTENDANCE_FILE):

        df = pd.read_csv(ATTENDANCE_FILE)

        st.dataframe(df)

    else:

        st.warning("No attendance records found")

# ---------------- AI CAMERA ---------------- #

elif choice == "AI Camera":

    st.subheader("📸 AI Camera Attendance System")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    # ---------------- FACE MESH ---------------- #

    mpFaceMesh = mp.solutions.face_mesh

    faceMesh = mpFaceMesh.FaceMesh(
        max_num_faces=2
    )

    mpDraw = mp.solutions.drawing_utils

    drawSpec = mpDraw.DrawingSpec(
        thickness=1,
        circle_radius=0
    )

    # ---------------- CAMERA ---------------- #

    camera = cv2.VideoCapture(0)

    while run:

        success, img = camera.read()

        if not success:

            st.error("Camera not working")

            break

        imgRGB = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # -------- FACE MESH -------- #

        results = faceMesh.process(imgRGB)

        if results.multi_face_landmarks:

            for faceLms in results.multi_face_landmarks:

                mpDraw.draw_landmarks(
                    img,
                    faceLms,
                    mpFaceMesh.FACEMESH_TESSELATION,
                    drawSpec,
                    drawSpec
                )

        # -------- FACE RECOGNITION -------- #

        facesCurFrame = face_recognition.face_locations(
            imgRGB
        )

        encodesCurFrame = face_recognition.face_encodings(
            imgRGB,
            facesCurFrame
        )

        for encodeFace, faceLoc in zip(
            encodesCurFrame,
            facesCurFrame
        ):

            # Face Distance

            faceDis = face_recognition.face_distance(
                encodeListKnown,
                encodeFace
            )

            matchIndex = np.argmin(faceDis)

            # Confidence Threshold

            if faceDis[matchIndex] < 0.50:

                name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = faceLoc

                # Green Box

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.rectangle(
                    img,
                    (x1, y2 - 35),
                    (x2, y2),
                    (0, 255, 0),
                    cv2.FILLED
                )

                cv2.putText(
                    img,
                    name,
                    (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                # Mark Attendance

                markAttendance(name)

            else:

                y1, x2, y2, x1 = faceLoc

                # Red Box

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.rectangle(
                    img,
                    (x1, y2 - 35),
                    (x2, y2),
                    (0, 0, 255),
                    cv2.FILLED
                )

                cv2.putText(
                    img,
                    "UNRECOGNIZED",
                    (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        FRAME_WINDOW.image(
            cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )
        )

    camera.release()

# ---------------- ANALYTICS ---------------- #

elif choice == "Analytics":

    st.subheader("📈 Attendance Analytics")

    if os.path.exists(ATTENDANCE_FILE):

        df = pd.read_csv(ATTENDANCE_FILE)

        counts = df['Name'].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            counts.index,
            counts.values
        )

        ax.set_xlabel("Students")

        ax.set_ylabel("Attendance Count")

        ax.set_title("Person Wise Attendance")

        st.pyplot(fig)

    else:

        st.warning("No attendance data available")