import os
import cv2
import time

name = input("Enter name of person: ")
face_dir = os.path.join('./dataset/train/', name)

if not os.path.isdir('./dataset'):
    os.mkdir('dataset')
    os.mkdir('./dataset/train')
    os.mkdir('./dataset/test')

if not os.path.isdir('./dataset/train'):
    os.mkdir('train')

print(face_dir)

if os.path.isdir(face_dir):
    for f in os.listdir(face_dir):
        os.remove(os.path.join(face_dir, f))
    os.rmdir(face_dir)

os.mkdir(face_dir)
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    './haarcascade/haarcascade_frontalface_default.xml')

print('press q to close the web cam')
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    curr_time = int(time.time() * 1000)

    for (x, y, w, h) in faces:
        cropped_face = frame[y:y+h, x:x+w]
        cv2.imwrite(f"{os.path.join(face_dir, str(curr_time))}.png",
                    cropped_face)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
