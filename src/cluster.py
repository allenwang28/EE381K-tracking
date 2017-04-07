from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2

if __name__ == "__main__":
    fileDir = os.path.dirname(os.path.realpath(__file__))
    vidDir = os.path.join(fileDir, '..', 'videos')

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
