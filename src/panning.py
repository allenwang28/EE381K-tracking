import cv2
import numpy as np
import os

import pandas as pd

import colors

fileDir = os.path.dirname(os.path.realpath(__file__))
vidDir = os.path.join(fileDir, '..', 'videos')

defaultVidPath = os.path.join(vidDir, 'gswokc-2.mp4')


class Panner:
    _cap = None
    _smoothedTrajectory = None

    def __init__(self, videoPath):
        self._cap = cv2.VideoCapture(videoPath)
        self._numFrames = int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self._fps = int(self._cap.get(cv2.cv.CV_CAP_PROP_FPS))

    def getPanningTrajectory(self):
        if self._smoothedTrajectroy is None:
            # Take first frame and find corners in it
            ret, old_frame = cap.read()
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)

            (h,w) = old_frame.shape[:2]

            last_T = None
            prev_to_cur_transform = []

            while(curr < self._numFrames - 1):
                curr += 1
                ret, frame = cap.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                prev_corner = []
                cur_corner = []

                for i, st in enumerate(st):
                    if st == 1:
                        prev_corner.append(p0[i])
                        cur_corner.append(p1[i])
                prev_corner = np.array(prev_corner)
                cur_corner = np.array(cur_corner)
                T = cv2.estimateRigidTransform(prev_corner, cur_corner, False)

                if T is None:
                    print "T is None"

                dx = T[0,2]
                dy = T[1,2]
                da = np.arctan2(T[1,0], T[0,0])
                prev_to_cur_transform.append([dx,dy,da])
                old_frame = frame[:]
                old_gray = frame_gray[:]

            prev_to_cur_transform = np.array(prev_to_cur_transform)
            trajectory = np.cumsum(prev_to_cur_transform, axis=0)
            trajectory = pd.DataFrame(trajectory)
            self._smoothedTrajectory = pd.rolling_mean(trajectory, window=30)
            self._smoothedTrajectory.fillna(method='bfill')
        return self._smoothedTrajectory

if __name__ == "__main__":
    cap = cv2.VideoCapture(defaultVidPath)
    curr = 0

    numFrames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 200,
                           qualityLevel = 0.01,
                           minDistance = 30.0,
                           blockSize = 3 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #old_gray_top = colors.get_top_image(old_gray)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    last_T = None
    prev_to_cur_transform = []

    while(curr < numFrames - 1):
        curr += 1
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        prev_corner = []
        cur_corner = []

        for i, st in enumerate(st):
            if st == 1:
                prev_corner.append(p0[i])
                cur_corner.append(p1[i])
        prev_corner = np.array(prev_corner)
        cur_corner = np.array(cur_corner)
        T = cv2.estimateRigidTransform(prev_corner, cur_corner, False)

        if T is None:
            print "T is None"

        dx = T[0,2]
        dy = T[1,2]
        da = np.arctan2(T[1,0], T[0,0])
        prev_to_cur_transform.append([dx,dy,da])
        old_frame = frame[:]
        old_gray = frame_gray[:]

    prev_to_cur_transform = np.array(prev_to_cur_transform)
    trajectory = np.cumsum(prev_to_cur_transform, axis=0)
    trajectory = pd.DataFrame(trajectory)
    smoothed_trajectory = pd.rolling_mean(trajectory, window=30)
    smoothed_trajectory.fillna(method='bfill')

    print len(smoothed_trajectory)
    print numFrames
    new_prev_to_cur_transform = prev_to_cur_transform + (smoothed_trajectory - trajectory)

    T = np.zeros((2,3))
    new_prev_to_cur_transform = np.array(new_prev_to_cur_transform)
    cap = cv2.VideoCapture(defaultVidPath)


    for k in range(numFrames - 1):
        ret, frame = cap.read()
        T[0,0] = np.cos(new_prev_to_cur_transform[k][2])
        T[0,1] = -np.sin(new_prev_to_cur_transform[k][2])
        T[1,0] = np.sin(new_prev_to_cur_transform[k][2])
        T[1,1] = np.cos(new_prev_to_cur_transform[k][2])
        T[0,2] = new_prev_to_cur_transform[k][0]
        T[1,2] = new_prev_to_cur_transform[k][1]
        frame2 = cv2.warpAffine(frame, T, (w,h))
        cv2.imshow('vid', frame2)
        cv2.waitKey(20)



