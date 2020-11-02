import cv2
# from mtcnn import MTCNN
# import os
# import numpy as np
# from numpy import asarray
# from datetime import datetime
# from PIL import Image, ImageDraw, ImageFont

path = 'known_faces'
# detector = MTCNN()

known_faces = []
names = []


# def encodingKnownFaces():
#     files = os.listdir(path)

#     for file in files:
#         img = face_recognition.load_image_file(f'{path}/{file}')
#         # imgResize = cv2.resize(img, None, fx=0.25, fy=0.25)
#         # imgEncode = face_recognition.face_encodings(img)[0]

#         # known_faces.append(imgEncode)
#         # names.append(os.path.splitext(file)[0])

#     print(names)

def get_faces(frame, size=(160, 160)):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    extracted_faces = []
    for face in faces:
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        frame = drawBoudingBox(frame, x1, x2, y1, y2)

        # extracted_face = frame[y1:y2, x1:x2]
        # img = Image.fromarray(extracted_face)
        # resize_img = img.resize(size)

        # face_arr = asarray(resize_img)        
        # extracted_faces.append(face_arr)

    # return np.array(extracted_faces)
    return 1

def drawBoudingBox(frame, x1, x2, y1, y2, name="unknown"):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (x1, y2), (x2, y2+40),
                  (0, 255, 0), cv2.FILLED)

    fontpath = "./font.ttf"
    font = ImageFont.truetype(fontpath, 30)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x1+5, y2), name, font=font, fill=(255, 255, 255))
    frame = np.array(img_pil)

    cv2.putText(frame,  None, (200, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main():

    cap = cv2.VideoCapture(0)

    while True:
        res, frame = cap.read()

        # extracted_faces = get_faces(frame)

        # for face in extracted_faces:

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
