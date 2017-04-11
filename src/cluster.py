from sklearn.cluster import DBSCAN

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2

import colors

fileDir = os.path.dirname(os.path.realpath(__file__))
vidDir = os.path.join(fileDir, '..', 'videos')
imgDir = os.path.join(fileDir, '..', 'images')

sampleImgPath = os.path.join(imgDir, 'test.jpg')

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 10:
                return True
            elif i==row1-1 and j==row2-1:
                return False

def clusterSegmentedImage(img):
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    for i,cnt1 in enumerate(contours):
        x = i    
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist = find_if_close(cnt1,cnt2)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    maximum = int(status.max())+1
    for i in xrange(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)
    cv2.fillPoly(img, pts=unified, color=(255,255,255))
    return img


def test1():
    GSW_AWAY_LOWER = (115, 190, 80)
    GSW_AWAY_UPPER = (135, 255, 150)
    GSW_AWAY = (GSW_AWAY_LOWER, GSW_AWAY_UPPER)
    img = cv2.imread(sampleImgPath)
    img = colors.get_crowdless_image(img)
    mask = colors.get_jersey_mask(img, GSW_AWAY_LOWER, GSW_AWAY_UPPER)

    img = clusterSegmentedImage(mask)

    cv2.imshow('img',img)
    cv2.waitKey(0)

def getCentroids(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    centroids = []

    for cnt in contours:
        if 300 < cv2.contourArea(cnt) < 10000:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append((cx,cy))
    return centroids


def centroidTest():
    img = cv2.imread(sampleImgPath)

    img = colors.get_crowdless_image(img)
    mask = colors.get_jersey_mask(img, (115, 190, 80), (125, 260, 260))

    print getCentroids(mask)

    """
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        if 50 < cv2.contourArea(cnt) < 5000:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(img, (cx, cy), 10, (0, 255, 255))
            #cv2.drawContours(img, [cnt], 0, (0,255,255), -1)

    cnt = contours[0]
    M = cv2.moments(cnt)
    print M
    """

    cv2.imshow('img', img)
    cv2.waitKey(0)

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
    #centroidTest()
    test1()


