from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2

from colors import get_crowdless_image

fileDir = os.path.dirname(os.path.realpath(__file__))
vidDir = os.path.join(fileDir, '..', 'videos')
imgDir = os.path.join(fileDir, '..', 'images')

sampleImgPath = os.path.join(imgDir, 'test.jpg')

def centroidTest():
    img = cv2.imread(sampleImgPath, 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    cnt = contours[0]
    M = cv2.moments(cnt)
    print M

def blobTest():
    img = cv2.imread(sampleImgPath)
    img = get_crowdless_image(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_img = get_crowdless_image(gray_img)

    detector = cv2.SimpleBlobDetector()

    keypoints = detector.detect(gray_img)

    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)


def clusterTest():

    #defaultVidPath = os.path.join(vidDir, 'housas-1.mp4')
    defaultVidPath = os.path.join(vidDir, 'gswokc-1.mp4')

    cap = cv2.VideoCapture(defaultVidPath)

    while(1):
        ret,frame = cap.read()

        if ret == True:
            labimg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            n = 0
            while(n<4):
                labimg = cv2.pyrDown(labimg)
                n = n+1

            feature_image=np.reshape(labimg, [-1, 3])
            rows, cols, chs = labimg.shape

            indices = np.dstack(np.indices(frame.shape[:2]))
            xycolors = np.concatenate((frame, indices), axis=-1) 
            np.reshape(xycolors, [-1,5])

            db = DBSCAN(eps=10, min_samples=50, metric = 'euclidean',algorithm ='auto')
            db.fit(feature_image)
            labels = db.labels_

            plt.figure(2)
            plt.subplot(2, 1, 1)
            plt.imshow(frame)
            plt.axis('off')
            plt.subplot(2, 1, 2)
            plt.imshow(np.reshape(labels, [rows, cols]))
            plt.axis('off')
            plt.show()

            cv2.imshow('img',frame)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            raw_input()
        else:
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    centroidTest()

