import cv2
from mtcnn.mtcnn import MTCNN

# initialize the face detector
detector = MTCNN()

# initialize the camera
camera = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)  

while True:
    # get a frame from the camera
    ret, frame = camera.read()

    # detect faces in the frame
    faces = detector.detect_faces(frame)

    # extract the bounding box from the faces
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # display the frame
    cv2.imshow("Camera", frame)

    # wait for a key event and end the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the camera
camera.release()

# close all windows
cv2.destroyAllWindows()