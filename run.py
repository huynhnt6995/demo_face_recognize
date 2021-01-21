from mtcnn import MTCNN
from cv2 import cv2

detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = detector.detect_faces(frame)
    for result in results:
        face_box = result['box']
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2] + face_box[0], face_box[3] + face_box[1]), (0, 0, 255), 1)
        cv2.putText(frame, str(result['confidence']), (face_box[0], face_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv2.imshow('output', frame)

    k = cv2.waitKey(1)
    if k==27:    # Esc key to stop
        break

