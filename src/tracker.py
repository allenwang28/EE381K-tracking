import numpy as np
import cv2
import os

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
TERMINATION_CRITERIA = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
class Tracker:
    _frame = None

    _row = None
    _height = None
    _column = None
    _width = None

    _roi = None

    def __init__(self, initialFrame, row, height, column, width):
        self._frame = initialFrame
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
    
    def setTrackWindow(self, (c, r, h, w)):
        self.setRow(r)
        self.setColumn(c)
        self.setHeight(h)
        self.setWidth(w)

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

    def getRoi(self):
        if self._roi is None:
            self._roi = self._frame[self.getRow():self.getRow() + self.getHeight(), self.getColumn(): self.getColumn() + self.getWidth()]
        return self._roi

    def getHsvRoi(self):
        return cv2.cvtColor(self.getRoi(), cv2.COLOR_BGR2HSV)

    def getMask(self):
#        return cv2.inRange(self.getHsvRoi(), np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        return cv2.inRange(self.getHsvRoi(), np.array((0., 0.,0.)), np.array((255.,255.,255.)))

    def getRoiHist(self):
        roi_hist = cv2.calcHist([self.getHsvRoi()],[0],self.getMask(),[180],[0,180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        return roi_hist

    # CamShift implementation
    def getPoints(self, hsv):
        try:
            print "Original trackWindow {}".format(self.getTrackWindow())
            dst = cv2.calcBackProject([hsv],[0], self.getRoiHist(),[0,180],1)
            ret, trackWindow = cv2.CamShift(dst, self.getTrackWindow(), TERMINATION_CRITERIA)
            trackWindow = containTrackingWindow(trackWindow)
            print "New trackWindow {}\n\n".format(trackWindow)
            self.setTrackWindow(trackWindow)
            pts = cv2.cv.BoxPoints(ret)
            return np.int0(pts)
        except Exception as e:
            print e
            return None

    def isLive(self):
        return not self.getTrackWindow() == (0, 0, 0, 0)

    def drawOnFrame(self, frame):
        x,y,w,h = self.getTrackWindow() 
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 255), 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pts = self.getPoints(hsv)
        if pts is None:
            """
            # if CamShift fails let's use MeanShift in the MEAN time
            dst = cv2.calcBackProject([hsv],[0], self.getRoiHist(),[0,180],1)
            ret, trackWindow = cv2.meanShift(dst, self.getTrackWindow(), TERMINATION_CRITERIA)
            self.setTrackWindow(trackWindow)
            x,y,w,h = trackWindow
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)
            """
            return frame
        cv2.polylines(frame, [pts], True, 255, 2)
        return frame

    """
    # This is me attempting Mean Shift
    def drawOnFrame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0], self.getRoiHist(),[0,180],1)
        ret, trackWindow = cv2.meanShift(dst, self.getTrackWindow(), TERMINATION_CRITERIA)
        self.setTrackWindow(trackWindow)
        x,y,w,h = trackWindow
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)
        return img
    """

# Trying to prevent the tracking window from becoming ridiculously large..
def containTrackingWindow(trackingWindow):
    x,y,w,h = trackingWindow
    if w > 50:
        w = 50
    if h > 50:
        h = 50
    return (x,y,w,h)

if __name__ == "__main__":
    fileDir = os.path.dirname(os.path.realpath(__file__))
    vidDir = os.path.join(fileDir, '..', 'videos')

    defaultVidPath = os.path.join(vidDir, 'housas-1.mp4')
    #defaultVidPath = os.path.join(vidDir, 'gswokc-1.mp4')

    cap = cv2.VideoCapture(defaultVidPath)
    # take first frame of the video
    ret,frame = cap.read()

    """
    # setup initial location of window
    r1,h1,c1,w1 = 350,5,150,5  # simply hardcoded the values
    track_window1 = (c1,r1,w1,h1)
    # set up the ROI for tracking
    roi1 = frame[r1:r1+h1, c1:c1+w1]
    hsv_roi1 =  cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_roi1, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist1 = cv2.calcHist([hsv_roi1],[0],mask1,[180],[0,180])
    cv2.normalize(roi_hist1,roi_hist1,0,255,cv2.NORM_MINMAX)
    """

    # Test Tracker
    
    trackers = []
    # housas-1
    trackers.append(Tracker(frame, 350, 5, 150, 5)) # Harden
    trackers.append(Tracker(frame, 380, 5, 400, 5)) # Green
    trackers.append(Tracker(frame, 525, 5, 780, 5)) # Anderson
    trackers.append(Tracker(frame, 395, 5, 1120, 5)) # Mills
    trackers.append(Tracker(frame, 450, 5, 780, 5)) # Kawhi
    trackers.append(Tracker(frame, 275, 5, 790, 5)) # Dedmon
    trackers.append(Tracker(frame, 275, 5, 680, 5)) # Capela
    trackers.append(Tracker(frame, 230, 5, 845, 5)) # Ariza
    """
    # gswokc-1
    trackers.append(Tracker(frame, 410, 5, 175, 5)) # Steph
    trackers.append(Tracker(frame, 410, 5, 390, 5)) # Roberson
    trackers.append(Tracker(frame, 410, 5, 1140, 5)) # Livingston
    trackers.append(Tracker(frame, 220, 5, 880, 5)) # Barnes
    trackers.append(Tracker(frame, 300, 5, 800, 5)) # Barnes
    """

    while(1):
        ret,frame = cap.read()

        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            """
            dst1 = cv2.calcBackProject([hsv],[0],roi_hist1,[0,180],1)
            # apply meanshift to get the new location
            ret1, track_window1 = cv2.CamShift(dst1, track_window1, term_crit)
            # Draw it on image
            pts1 = cv2.cv.BoxPoints(ret1)
            pts1 = np.int0(pts1)
            cv2.polylines(frame,[pts1],True, 255,2)
            """

            for tracker in trackers:
                tracker.drawOnFrame(frame)

            print("\n\n")
            cv2.imshow('img',frame)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            raw_input()
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
