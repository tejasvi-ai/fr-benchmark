import cv2, face_recognition
from glob import glob
from collections import defaultdict
import pickle


def getRep(f):
    image = cv2.imread(f)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="cnn")
    encodings = face_recognition.face_encodings(rgb, boxes)
    return encodings


dirs = ["new_flex_milpitas_nested", "milpitas_nested", "celeb"]

for directory in dirs:
    f_list = glob(directory + "/*/*")
    d = defaultdict(list)

    for i, f in enumerate(f_list):
        print(
            "\r{directory} {per}".format(
                directory=directory, per=round(float(i) / len(f_list) * 100, 2)
            ),
            end="",
        )
        embd = getRep(f)
        if embd:
            d[f.split("/")[-2]].append((embd, f))
    with open(directory + "_dlib.pkl", "wb") as file:
        pickle.dump(d, file)
