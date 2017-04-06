import numpy as np
import cv2
import os

class Tracker:
    _frame = None

    _row = None
    _height = None
    _column = None
    _width = None

    def __init__(self, frame, row, height, column, width):
        self._frame = frame
        self._row = row
        self._height = height
        self._column = column
        self._width = width

    def setRow(self, row):
        self._row = row

    def setHeight(self, height):
        self._height = height

    def setColumn(self, column):
        self._column = column

    def setWidth(self, width):
        self._width = width

    def getRow(self):
        return self._row

    def getHeight(self):
        return self._height

    def getColumn(self):
        return self._column

    def getWidth(self):
        return self._width

    def getTrackWindow(self):
        return (self._column, self._row, self._height, self._width)


if __name__ == "__main__":
    fileDir = os.path.dirname(os.path.realpath(__file__))
    vidDir = os.path.join(fileDir, '..', 'videos')

    defaultVidPath = os.path.join(vidDir, 'housas-1.mp4')

    cap = cv2.VideoCapture(defaultVidPath)
    # take first frame of the video
    ret,frame = cap.read()
    # setup initial location of window
    #r,h,c,w = 250,90,400,125  # simply hardcoded the values
    r,h,c,w = 350,5,150,5  # simply hardcoded the values
    track_window = (c,r,w,h)
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    while(1):
        ret,frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

            # apply meanshift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image

            print track_window

            pts = cv2.cv.BoxPoints(ret)
            pts = np.int0(pts)

            cv2.polylines(frame,[pts],True, 255,2)

            cv2.imshow('img2',frame)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k)+".jpg",frame)
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
