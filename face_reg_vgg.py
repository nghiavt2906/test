import cv2, os, joblib
import numpy as np
from time import time
from imutils.video import WebcamVideoStream
from numpy import asarray, expand_dims
from PIL import Image, ImageDraw, ImageFont
from fast_mtcnn import FastMTCNN
from sklearn.preprocessing import Normalizer,  LabelEncoder
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from sklearn.svm import SVC

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=0.5,
    margin=14,
    factor=0.6,
    keep_all=True,
    device='cuda'
)

embedding_model = VGGFace(
    model='resnet50', 
    include_top=False, 
    input_shape=(224, 224, 3), 
    pooling='avg'
)


def draw_rect(frame, box):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
    x1, y1, x2, y2 = np.float32(x1), np.float32(y1), np.float32(x2), np.float32(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def set_label(frame, box, text):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
    x1, y1, x2, y2 = np.float32(x1), np.float32(y1), np.float32(x2), np.float32(y2)

    height = np.float32(y2 + 40)

    cv2.rectangle(frame, (x1, y2), (x2, height), (0, 255, 0), cv2.FILLED)

    fontpath = "./font.ttf"
    font = ImageFont.truetype(fontpath, 30)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x1+5, y2), text, font=font, fill=(255, 255, 255))
    frame = np.array(img_pil)

    return frame

def getTrainDataFromDir(size=(224, 224)):
    path = 'images/train_data'
    extracted_faces = []
    labels = []

    for filename in os.listdir(path):
        img = Image.open(f'{path}/{filename}')
        img = img.convert('RGB')
        img_arr = asarray(img)
        boxes, faces = fast_mtcnn([img_arr])

        face = faces[0]
        img = Image.fromarray(face)
        img_resize = img.resize(size)
        face_arr = asarray(img_resize)

        extracted_faces.append(face_arr)

        label = filename.split('_')[0]
        labels.append(label)  

    return asarray(extracted_faces), asarray(labels)

# def process_faces(faces, size=(224, 224)):
#     extracted_faces = []

#     for face in faces:
#         arr = asarray(face)
#         if arr.shape[0] > 0 and arr.shape[1] > 0:
#             img = Image.fromarray(face)
#             img_resize = img.resize(size)
#             face_arr = asarray(img_resize)

#             extracted_faces.append(face_arr)
    
#     return extracted_faces

def get_embedding(embedding_model, face):
    pixels = face.astype('float32')
    samples = expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    yhat = embedding_model.predict(samples)

    return yhat[0]

def get_models(embedding_model, isTrain):
    X_train, y_train = getTrainDataFromDir()
    embedding_X_train = list()
    path = 'models/vgg_models'
    svc_filename, encoder_filename = 'svc_model.mdl', 'encoder_model.mdl'

    encoder_path = os.path.join(path, encoder_filename)
    svc_path = os.path.join(path, svc_filename)

    if isTrain == 'n' and os.path.exists(svc_path):
        print('Loading the model from dir...')
        encoder = joblib.load(encoder_path)
        svc_model = joblib.load(svc_path)
        return svc_model, encoder

    for face in X_train:
        embedding = get_embedding(embedding_model, face)
        embedding_X_train.append(embedding)

    embedding_X_train = asarray(embedding_X_train)

    norm = Normalizer(norm='l2')
    trainX = norm.transform(embedding_X_train)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    trainy = encoder.transform(y_train)

    print('Training the model...')

    svc_model = SVC(kernel='sigmoid', probability=True)
    svc_model.fit(trainX, trainy)

    joblib.dump(svc_model, svc_path)
    joblib.dump(encoder, encoder_path)

    return svc_model, encoder

def main():
    isTrain = ""
    while isTrain != 'y' and isTrain != 'n':
        isTrain = input("Train the model or not (y/n)? ")
    
    # cap = cv2.VideoCapture(0)
    cap = WebcamVideoStream(src=0).start()

    svc_model, encoder = get_models(embedding_model, isTrain)
    print('Completed')

    start = time()
    count = 0
    fps = 0

    while True:
        frame = cap.read()


        if time() - start >= 1:
            fps = count
            count = 0
            start = time()

        count += 1

        boxes, faces = fast_mtcnn([frame])


        if len(faces):
            extracted_faces = []
            
            for face in faces:
                arr = asarray(face)
                if arr.shape[0] > 0 and arr.shape[1] > 0:
                    img = Image.fromarray(face)
                    img_resize = img.resize((224, 224))
                    face_arr = asarray(img_resize)

                    extracted_faces.append(face_arr)

            if len(extracted_faces) == 0:
                continue
            
            embedding_X_train = []

            for face in extracted_faces:
                embedding = get_embedding(embedding_model, face)
                embedding_X_train.append(embedding)

            # print(len(embedding_X_train[0]))

            embedding_X_train = asarray(embedding_X_train)

            norm = Normalizer(norm='l2')
            trainX = norm.transform(embedding_X_train)

            preds = svc_model.predict(trainX)
            pred_probs = svc_model.predict_proba(trainX)
            pred_names = encoder.inverse_transform(preds)

            for pred_prob, pred_name, box in zip(pred_probs, pred_names, boxes[0]):
                accuracy = pred_prob[np.argmax(pred_prob)]*100

                if accuracy > 75:
                    text = '{} {:.2f}%'.format(pred_name, accuracy)
                else:
                    text = 'Unknown'

                draw_rect(frame, box)
                frame = set_label(frame, box, text)

            for box in boxes[0]:
                draw_rect(frame, box)

            for box in boxes[0]:
                draw_rect(frame, box)

        frame = cv2.putText(frame, 'FPS: {}'.format(fps), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Face recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.stop()
            break

if __name__ == "__main__":
    main()