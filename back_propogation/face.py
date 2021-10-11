import os
import numpy as np
import cv2
import random
import pickle
import time
import sys


def create(name):
    path = os.getcwd()
    print(f"creating data for {name}..")
    path = os.path.join(path, "dataset")
    if os.path.isdir(path) == False:
        os.mkdir(path)
    filefolder = os.path.join(path, name)
    try:
        os.mkdir(filefolder, 0o755)
    except OSError:
        print("Name has already been taken")
        return
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    for i in range(0, 25):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi_gray, (200, 200))
            cv2.imwrite(os.path.join(filefolder, 'image_%i.jpg' % i), roi)
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
    if len(os.listdir(filefolder)) == 0:
        os.rmdir(filefolder)
    cv2.destroyAllWindows()
    print(f"created..")


def update():
    print("updating the dataset..")

    path = os.getcwd()
    category = []
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    Data_Dir = os.path.join(path, 'dataset')
    training = []
    for file in os.listdir(Data_Dir):
        img_dir = os.path.join(Data_Dir, file)
        category.append(file)
        for images in os.listdir(img_dir):
            img = cv2.imread(os.path.join(img_dir, images),
                             cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150, 150))
            training.append([img, category.index(file)])
    random.shuffle(training)
    x = []
    y = []
    for features, labels in training:
        x.append(features)
        y.append(labels)
    x = np.array(x).reshape(-1, 150, 150, 1)
    pickle_out = open("training_img.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()
    pickle_out = open("training_roll_no.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    recognizer.train(x, np.array(y))
    recognizer.save("trainer.yml")
    print("updated..")


def recognise():
    print("starting recogniser...")
    path = os.getcwd()
    path = os.path.join(path, "dataset")
    Categories = []
    for files in os.listdir(path):
        Categories.append(files)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer1 = cv2.face.FisherFaceRecognizer_create()
    pickle_in = open("training_img.pickle", "rb")
    x = pickle.load(pickle_in)
    pickle_in = open("training_roll_no.pickle", "rb")
    y = pickle.load(pickle_in)
    recognizer.read("trainer.yml")
    face_cascade = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    start = time.time()
    while (cv2.waitKey(1) & 0xFF == ord('q')) == False:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (150, 150))
            y1, conf = recognizer.predict(roi_gray)
            print(Categories[y1])
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame, Categories[y1], (x, y), font, 1, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.imshow('img', frame)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = sys.argv
    method = args[1]
    method_dict = {
        "update": update,
        "recognise": recognise,
    }
    if method == "create":
        create(args[2])
    else:
        method_dict[method]()
