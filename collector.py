import cv2
from os.path import join
from time import sleep

name = input("Name: ")

cap = cv2.VideoCapture(0)
path = "images/train_data"
iters = 100

print('Start capturing now...')

for i in range(1, iters+1):
    _, frame = cap.read()

    filename = "{}_{}.jpg".format(name, i)
    cv2.imwrite(join(path, filename), frame)

    print("Captured for image {}".format(i))
    # sleep(0.08)

print("Completed")
