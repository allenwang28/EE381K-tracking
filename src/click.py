import cv2
import os


def mouseCallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print "{}, {}".format(x, y)

if __name__ == "__main__":
    fileDir = os.path.dirname(os.path.realpath(__file__))
    vidDir = os.path.join(fileDir, '..', 'videos')
    imgDir = os.path.join(fileDir, '..', 'images')

    defaultVidPath = os.path.join(vidDir, 'gswokc-5.mp4')
    defaultImgPath = os.path.join(imgDir, 'test.jpg')

    img = cv2.imread(defaultImgPath)
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", mouseCallback)

    cap = cv2.VideoCapture(defaultVidPath)
    ret, frame = cap.read()

    cv2.imshow("frame", frame)
    cv2.waitKey(0)


