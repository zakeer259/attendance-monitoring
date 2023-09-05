import cv2
import numpy as np
import mtcnn
import pandas as pd
from Nitheesh.model import architecture
from training import normalize, l2_normalizer
from scipy.spatial.distance import cosine
import pickle
import os
from datetime import datetime


CONFIDENCE_THRESHOLD = 0.99
RECOGNITION_THRESHOLD = 0.30
REQUIRED_SIZE = (160, 160)


def get_face(image, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def get_encode(encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict



def mark_attendance(name):
    print(f"Marking attendance for {name}")
    now = datetime.now()
    datetime_string = now.strftime('%H:%M:%S')
    attendance_df = pd.read_csv("Attendance.csv", index_col="Names")
    if name not in attendance_df.index:
        new_row = pd.DataFrame({'Names': [name],
                                'ID': [str(len(attendance_df.index))],
                                'Intime': [datetime_string],
                                'Outtime': [""]})
        new_row.to_csv('Attendance.csv', mode='a', index=False, header=False)
    else:
        attendance_df['Outtime'] = attendance_df['Outtime'].astype(str)
        attendance_df.at[name, 'Outtime'] = datetime_string
        attendance_df.to_csv("Attendance.csv")
        print("Attendance marked for", name)


def detect(image, detector, encoder, encoding_dict):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image_rgb)
    for res in results:
        if res['confidence'] < CONFIDENCE_THRESHOLD:
            continue
        face, pt1, pt2 = get_face(image_rgb, res['box'])
        encode = get_encode(encoder, face, REQUIRED_SIZE)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < RECOGNITION_THRESHOLD and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(image, name, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(image, name + f'__{distance:.2f}', (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
            
            mark_attendance(name)

    return image



import time

if __name__ == "__main__":
    required_shape = (220,220)
    face_encoder = architecture()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    record_duration = 60 # seconds

    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND") 
            break
        
        frame= detect(frame , face_detector , face_encoder , encoding_dict)

        cv2.imshow('camera', frame)

        if time.time() - start_time >= record_duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    


