# Library imports
import cv2
import numpy as np
import pickle
import itertools

# Local imports
import colors
import top_line_detection as tld
import hough
from geometry import get_intersection
import cluster

import homography
from homography import NbaCourt as nba

class FrameObject():
    # Variables
    _frameNum = None
    _videoTitle = None
    # Images
    _bgrImg = None
    _grayMask = None
    _dominantColorset = None
    _grayFlooded2 = None
    _awayColors = None
    _awayMask = None
    _homeColors = None
    _homeMask = None
    # Lines
    _sideline = None
    _baseline = None
    _freethrowline = None
    _closepaintline = None
    # Trackers
    _trackers = None

    _side = None

    # Booleans
    _sidelineDetected = True
    _baselineDetected = True
    _freethrowlineDetected = True
    _closepaintlineDetected = True

    # Points
    _sidelineBaseline = None # The far one in the corner
    _closepaintBaseline = None # Intersection between close paintline and baseline
    _closepaintFreethrow = None # Int btw close paintline and freethrow line
    _sidelineFreethrow = None # Int btw far sideline and freethrow line

    _pts = None

    _awayMaskCentroids = None
    _homeMaskCentroids = None

    _homography = None

    _verbose = True

    def __init__(self, img, frameNum, videoTitle, awayColors, homeColors, side, verbose = True):
        assert img is not None
        if side not in ['left', 'right']:
            raise Exception("Invalid side provided: {}".format(side))
        self._side = side
        self._bgrImg = img
        self._frameNum = frameNum
        self._videoTitle = videoTitle
        self._awayColors = awayColors
        self._homeColors = homeColors
        self._verbose = verbose


    # Exported methods
    def getGrayMask(self):
        if self._grayMask is None:
            d_c = self.getDominantColorset()
            self._grayMask = \
                colors.create_court_mask(self.getBgrImg(), d_c, True)
        return self._grayMask.copy()


    def getBgrImg(self):
        return self._bgrImg.copy()


    def getDominantColorset(self):
        if self._dominantColorset is None:
            self._dominantColorset = colors.get_dominant_colorset(self.getBgrImg())
        return self._dominantColorset.copy()


    def getGrayFlooded2(self):
        if self._grayFlooded2 is None:
            self._grayFlooded2 = \
                colors.get_double_flooded_mask(self.getGrayMask())
        return self._grayFlooded2.copy()

    def getSideline(self):
        if self._sideline is None:
            lines = tld.find_top_boundary(self.getGrayMask())
            # img = colors.gray_to_bgr(self._grayMask.copy())
            # hough.put_lines_on_img(img, lines)
            # cv2.imwrite('images/6584_toplines.jpg', img)
            if len(lines) < 2:
                self._baselineDetected = False
<<<<<<< HEAD
                #raise Exception("Baseline not detected")
                if self._verbose:
                    print("Baseline not detected")
=======
                raise Exception("Baseline not detected")
                #print("Baseline not detected")
>>>>>>> parent of 384e592... quick updates to show all possible points
                #return None
            self._sideline = lines[0]
            self._baseline = lines[1]
            if self._verbose:
                print "Sideline: {}".format(self._sideline)
                print "Baseline: {}".format(self._baseline)
        return self._sideline


    def getBaseline(self):
        if self._baseline is None:
            _ = self.getSideline()
        return self._baseline


    def getFreethrowline(self):
        if self._freethrowline is None:
            lines = hough.get_lines_from_paint(self.getGrayFlooded2(),
                self.getSideline(), self.getBaseline(), verbose=True)
            if lines[0] is None:
                self._freethrowlineDetected = False
<<<<<<< HEAD
                self._freethrowline = None
                if self._verbose:
                    print "Freethrow not detected"
            else:
                self._freethrowline = lines[0]
                if self._verbose:
                    print "Freethrow: {}".format(self._freethrowline)
            if lines[1] is None:
                self._closepaintlineDetected = False
                self._closepaintline = None
                if self._verbose:
                    print "Close paint not detected"
            else:
                self._closepaintline = lines[1]
                if self._verbose:
                    print "Close paint: {}".format(self._closepaintline)
=======
            if lines[1] is None:
                self._closepaintlineDetected = False
            self._freethrowline, self._closepaintline = lines
>>>>>>> parent of 384e592... quick updates to show all possible points
        return self._freethrowline


    def getClosepaintline(self):
        if self._closepaintline is None:
            _ = self.getFreethrowline()
        return self._closepaintline


    def getQuadranglePoints(self):
        if self._pts is None:
            try:
                self._pts = []
                self._pts.append(get_intersection(self.getSideline(), self.getFreethrowline()))
                self._pts.append(get_intersection(self.getSideline(), self.getBaseline()))
                self._pts.append(get_intersection(self.getClosepaintline(), self.getBaseline()))
                self._pts.append(get_intersection(self.getClosepaintline(), self.getFreethrowline()))
            except:
                self._pts = None
        return self._pts


    def getAwayJerseyMask(self):
        if self._awayMask is None:
            crowdlessImg = colors.get_crowdless_image(self.getBgrImg())
            self._awayMask = colors.get_jersey_mask(crowdlessImg, self._awayColors[0], self._awayColors[1])
            self._awayMask = cluster.clusterSegmentedImage(self._awayMask)
        return self._awayMask


    def getHomeJerseyMask(self):
        if self._homeMask is None:
            crowdlessImg = colors.get_crowdless_image(self.getBgrImg())
            self._homeMask = colors.get_jersey_mask(crowdlessImg, self._homeColors[0], self._homeColors[1])
            self._homeMask = cluster.clusterSegmentedImage(self._homeMask)
        return self._homeMask


    def getAwayMaskCentroids(self):
        if self._awayMaskCentroids is None:
            self._awayMaskCentroids = cluster.getCentroids(self.getAwayJerseyMask())
            self._awayMaskCentroids = colors.remap_from_crowdless_coords(self.getBgrImg(), self._awayMaskCentroids)
        return self._awayMaskCentroids

    def getHomeMaskCentroids(self):
        if self._homeMaskCentroids is None:
            self._homeMaskCentroids = cluster.getCentroids(self.getHomeJerseyMask())
            self._homeMaskCentroids = colors.remap_from_crowdless_coords(self.getBgrImg(), self._homeMaskCentroids)
        return self._homeMaskCentroids

    def getNumAwayMaskCentroids(self):
        return len(self.getAwayMaskCentroids())

    def getNumHomeMaskCentroids(self):
        return len(self.getHomeMaskCentroids())

    def setQuadranglePts(self, quadranglePts):
        self._pts = quadranglePts

    def freethrowlineDetected(self):
        return self._freethrowlineDetected

    def baselineDetected(self):
        return self._baselineDetected

    def sidelineDetected(self):
        return self._sidelineDetected

    def closepaintlineDetected(self):
        return self._closepaintlineDetected

    def allLinesDetected(self):
        return self._freethrowlineDetected and \
               self._baselineDetected and \
               self._closepaintlineDetected and \
               self._sidelineDetected

    def getFrameNumber(self):
        return self._frameNum

    def getVideoTitle(self):
        return self._videoTitle

    def getUndetectedLines(self):
        undetected_lines = ""
        if not self._freethrowlineDetected:
            undetected_lines += "free throw\n"
        if not self._baselineDetected:
            undetected_lines += "baseline\n"
        if not self._closepaintlineDetected:
            undetected_lines += "paintline\n"
        if not self._sidelineDetected:
            undetected_lines += "sideline\n"
        return undetected_lines

    def isAnchorPoint(self):
        return self.getNumAwayMaskCentroids() == 5 and \
               self.getNumHomeMaskCentroids() == 5 and \
               self.allLinesDetected()

    # This function should get rid of all of the "Nones"
    def generate(self):
        self.getQuadranglePoints()
        self.getAwayMaskCentroids()
        self.getHomeMaskCentroids()

    def drawLines(self, img=None):
        if img is None:
            img = self.getBgrImg()
        lines = [self.getFreethrowline(), self.getClosepaintline(),
            self.getSideline(), self.getBaseline()]
        if not self.allLinesDetected():
<<<<<<< HEAD
            #raise Exception("Not all lines were detected. Undetected: {}".format(self.getUndetectedLines()))
            if self._verbose:
                print("Not all lines were detected. Undetected: {}".format(self.getUndetectedLines()))
=======
            raise Exception("Not all lines were detected. Undetected: {}".format(self.getUndetectedLines()))
            #print("Not all lines were detected. Undetected: {}".format(self.getUndetectedLines()))
>>>>>>> parent of 384e592... quick updates to show all possible points
        #img = colors.gray_to_bgr(self.getGrayFlooded2())
        lines = [line for line in lines if line is not None]
        hough.put_lines_on_img(img, lines)
        return img

    def drawPoints(self, img=None):
        if img is None:
            img = self.getBgrImg()
        points = self.getQuadranglePoints()
<<<<<<< HEAD
        # Draw the points we were able to detect
        if self._verbose:
            print "Before"
            print points
        points = [point for point in points if point is not None]
        if self._verbose:
            print "After"
            print points
=======
        if points is None:
            return img
>>>>>>> parent of 384e592... quick updates to show all possible points
        hough.put_points_on_img(img, points)
        return img

    def drawAwayMaskCentroids(self, img=None):
        if img is None:
            img = self.getBgrImg()
        coordinates = self.getAwayMaskCentroids()

        circleColor = colors.hsv_to_bgr_color(self._homeColors[1])
        #circleColor = colors.hsv_to_bgr_color(self._awayColors[0])
        for (x,y) in coordinates:
            circs = cv2.circle(img, (x,y), 5, circleColor, -1)
        return img

    def drawHomeMaskCentroids(self, img=None):
        if img is None:
            img = self.getBgrImg()
        coordinates = self.getHomeMaskCentroids()
        circleColor = colors.hsv_to_bgr_color(self._awayColors[1])
        #circleColor = colors.hsv_to_bgr_color(self._homeColors[0])
        for (x,y) in coordinates:
            circs = cv2.circle(img, (x,y), 5, circleColor, -1)
        return img

    def showFrame(self):
        cv2.imshow('frame', self.getBgrImg())

    def showLines(self):
        cv2.imshow('lines', self.drawLines())

    def showPoints(self):
        cv2.imshow('points', self.drawPoints())

    def showAwayMaskCentroids(self):
        cv2.imshow('away players', self.drawAwayMaskCentroids())

    def showHomeMaskCentroids(self):
        cv2.imshow('home players', self.drawHomeMaskCentroids())

    def showHomeJerseyMask(self):
        cv2.imshow('home jersey mask', self.getHomeJerseyMask())

    def showAwayJerseyMask(self):
        cv2.imshow('away jersey mask', self.getAwayJerseyMask())

    # This function shows the mask centroids and the 
    # court lines
    def show(self):
        try:
            frame = self.getBgrImg()
            frame = self.drawHomeMaskCentroids(img=frame)
            frame = self.drawAwayMaskCentroids(img=frame)
            frame  = self.drawPoints(img=frame)
        except Exception as e:
            if self._verbose:
                print e
            pass
        cv2.imshow('frame-info', frame)

    def getHomography(self):
        if self._homography is None:
            # Quadrangle points are in this order:
            # 1. sideline freethrowline
            # 2. sideline baseline
            # 3. closepaintline baseline
            # 4. closepaintline freethrowline
            try:
                calcPoints = self.getQuadranglePoints()
                realPoints = [
                    nba.get_loc_coords('sideline, freethrow',self._side),
                    nba.get_loc_coords('sideline, baseline',self._side),
                    nba.get_loc_coords('closepaint, baseline',self._side),
                    nba.get_loc_coords('closepaint, freethrow',self._side),
                ]
                calcPoints = np.array(calcPoints).astype(float)
                realPoints = np.array(realPoints).astype(float)
                h, status = cv2.findHomography(calcPoints, realPoints)
                self._homography = h
            except Exception as e:
                return None
        return self._homography

    def showHomography(self):
        img = self.getBgrImg()
        h = self.getHomography()
        im_out = cv2.warpPerspective(img, h, (img.shape[0], 1500))
        cv2.imshow('homography', im_out)


def testlines(img_obj, save_filename):
    lines = [img_obj.getFreethrowline(), img_obj.getClosepaintline(),
        img_obj.getSideline(), img_obj.getBaseline()]
    img = colors.gray_to_bgr(img_obj.getGrayFlooded2())
    hough.put_lines_on_img(img, lines)
    cv2.imshow('lines', img)


def testpoints(img_obj, save_filename):
    lines = [img_obj.getFreethrowline(), img_obj.getClosepaintline(),
        img_obj.getSideline(), img_obj.getBaseline()]
    img = img_obj.getBgrImg()
    hough.put_lines_on_img(img, lines)
    points = img_obj.getQuadranglePoints()
    hough.put_points_on_img(img, points)
    cv2.imwrite(save_filename, img)


if __name__ == '__main__':
    import os
    fileDir = os.path.dirname(os.path.realpath(__file__))
    imgDir = os.path.join(fileDir, '..', 'images')

    sampleImgPath = os.path.join(imgDir, 'test.jpg')

    img = cv2.imread(sampleImgPath)
    
    fo = FrameObject(img, 0, '', colors.GSW_AWAY, colors.OKC_HOME, 'left')
    fo.showHomography()


    cv2.waitKey(0)


    


