import errno
import os

import argparse
import numpy as np
import cv2

import frame_object as fo
import colors

import panning
from panning import Panner

from tracker import Tracker

from homography import NbaCourt as nba

import pickle

import math


fileDir = os.path.dirname(os.path.realpath(__file__))
vidDir = os.path.join(fileDir, '..', 'videos')
dataPath = os.path.join(fileDir, '..', 'data')

#defaultVidPath = os.path.join(vidDir, 'housas-1.mp4')
#defaultVidPath = os.path.join(vidDir, 'gswokc-5.mp4')
#defaultVidPath = os.path.join(vidDir, 'gswokc-6.mp4')
defaultVidPath = os.path.join(vidDir, 'gswokc-8.mp4')
#defaultVidPath = os.path.join(vidDir, 'gswokc-4.mp4')

# NOTE - this is required because we only found the 
# ranges for a limited number of teams
ACCEPTED_AWAY_TEAMS = [
    'GSW',
    'HOU'
]

ACCEPTED_HOME_TEAMS = [
    'OKC',
    'SAS',
]

        
AWAY_TEAM_COLOR_DICT = {
    'GSW': colors.GSW_AWAY,
    'HOU': colors.HOU_AWAY,
}

HOME_TEAM_COLOR_DICT = {
     'OKC': colors.OKC_HOME,
     'SAS': colors.SAS_HOME,
}


# help/utility functions

# Finds the index of the first element that isn't None
# return None if the entire list is none
def notNoneIdx(L):
    for i, element in enumerate(L):
        if element is not None:
            return i
    return None

"""
def getAllVelocities(last, this):
    #print "=== # Get all velocities # ==="
    assert len(last) == len(this)
    velocities = []
    for lastPos, thisPos in zip(last.getQuadranglePoints(), this.getQuadranglePoints()):
        print "lastPos: {}, thisPos: {}".format(lastPos, thisPos)
        if lastPos is None or thisPos is None:
            velocities.append(None)
        else:
            velocities.append((lastPos[0] - thisPos[0], lastPos[1] - thisPos[1]))
    return velocities
"""
def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def getClosestPlayerPts(lastPts, thisPts, thresh = 50):
    newPts = []

    for lastPt in lastPts:
        # Setting minDist to thresh means that if the point moves more than thresh
        # Euclidean pixels then we won't count that as the closest point
        minDist = thresh
        closestPt = None
        for thisPt in thisPts:
            distance = dist(lastPt, thisPt)
            if distance < minDist:
                minDist = distance
                closestPt = thisPt
        if closestPt is not None:
            thisPts.remove(closestPt)
        newPts.append(closestPt)    
    return newPts 


def updatedCourtPoints(lastPts, thisPts):
    #print "=== # Getting average velocity # ==="
    assert len(lastPts) == len(thisPts)
    xVelocities = []
    yVelocities = []
    avgVelocity = None

    # First calculate the average velocity
    for lastPt, thisPt in zip(lastPts, thisPts):
        #print "lastPt: {}, thisPt: {}".format(lastPt, thisPt)
        if lastPt is None or thisPt is None:
            continue
        else:
            xDiff = thisPt[0] - lastPt[0]
            yDiff = thisPt[1] - lastPt[1]
            xVelocities.append(xDiff)
            yVelocities.append(yDiff)
    if len(xVelocities) == 0 or len(yVelocities) == 0:
        avgVelocity = [0,0]
    else:
        avgVelocity = [sum(xVelocities) / len(xVelocities), sum(yVelocities) / len(yVelocities)]
    #print "Average velocity: {}".format(avgVelocity)
    if abs(avgVelocity[0]) > 10:
        avgVelocity[0] = (avgVelocity[0] / abs(avgVelocity[0])) * 2
    if abs(avgVelocity[1]) > 10:
        avgVelocity[1] = (avgVelocity[1] / abs(avgVelocity[1])) * 2

    newPts = []
    # newPts = [(lastPt[0] + avgVelocity[0],lastPt[1] + avgVelocity[1]) for lastPt in lastPts]

    # For any point that is very far away or one that doesn't exist, we'll update it with the average velocity
    for lastPt, thisPt in zip(lastPts, thisPts):
        if thisPt is None or dist(lastPt, thisPt) > 10:
            newPts.append((lastPt[0] + avgVelocity[0], lastPt[1] + avgVelocity[1]))
        else:
            newPts.append(thisPt)
    return avgVelocity, newPts

# center = (640, 720) # gswokc-1
def getLikelyPlayerPoints(points, center = (800, 200)):
    # The most likely players are going to be closest to the center of the image.
    # Therefore we'll get the 5 closest points to the center
    print "Testing likely player points"
    print points
    points.sort(key=lambda pt: dist(pt, center), reverse=False)
    print points
    return points[:5]


def updatedPlayerPoints(lastPts, thisPts, avgVelocity, thresh = 60):
    # lastPts should be at least 5
    assert len(lastPts) >= 5

    if len(lastPts) > 5:
        lastPts = getLikelyPlayerPoints(lastPts)

    # First get the 5 points that are closest to our existing points
    thisPts = getClosestPlayerPts(lastPts, thisPts, thresh)

    newPts = []

    for lastPt, thisPt in zip(lastPts, thisPts):
        if thisPt is None:
            newPt = (int(lastPt[0] + avgVelocity[0]), int(lastPt[1] + avgVelocity[1]))
            newPts.append(newPt)
        else: 
            newPts.append(thisPt)
    return newPts


class Engine:
    # Video/opencv related variables
    _video = None # This is a path
    _cap = None
    _capLength = None
    _side = None

    # Lists
    _frameObjects = None
    _awayTrackers = None

    # Booleans
    _verbose = True

    # Game specific variables
    _awayTeam = None
    _homeTeam = None
    _awayColors = None
    _homeColors = None

    _panningTrajectory = None
    _homographies = None
    _courtPoints = None

    _awayPlayerPositions = None
    _homePlayerPositions = None

    _homeTrajectories = None
    _awayTrajectories = None

    _frameObjectsPath = None
    _panningTrajectoriesPath = None

    _trackers = None
    _avgVelocities = None


    def __init__(self, video, awayTeam, homeTeam, side, verbose = True):
        self._video = video
        if awayTeam not in ACCEPTED_AWAY_TEAMS:
            raise Exception("Invalid away team provided: {}".format(awayTeam))
        if homeTeam not in ACCEPTED_HOME_TEAMS:
            raise Exception("Invalid home team provided: {}".format(homeTeam))
        if side not in ['left', 'right']:
            raise Exception("Invalid side provided: {}".format(side))
        self._awayTeam = awayTeam
        self._homeTeam = homeTeam
        self._verbose = verbose
        self._side = side
        self._cap = cv2.VideoCapture(video)
        self._capLength = int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        self._videoTitle = os.path.splitext(os.path.basename(video))[0]

        self._frameObjectsPath = os.path.join(dataPath, '{}-frames.p'.format(self._videoTitle))
        self._filledPtsPath = os.path.join(dataPath, '{}-filledPts.p'.format(self._videoTitle))
        self._avgVelocitiesPath = os.path.join(dataPath, '{}-avgVelocities.p'.format(self._videoTitle))

        self._panningTrajectoriesPath = os.path.join(dataPath, '{}-traj.p'.format(self._videoTitle))

        self._width = int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))


        self._fourcc = cv2.cv.CV_FOURCC(*'X264')

        if verbose:
            print(("#===== Opening {} =====#".format(video)))
            print(("Successful?: {}".format(str(self._cap.isOpened()))))
            print(("Frame width: {}".format(self._width)))
            print(("Frame height: {}".format(self._height)))
            print(("FPS: {}".format(str(self._cap.get(cv2.cv.CV_CAP_PROP_FPS)))))
            print(("Frame count: {}".format(self._capLength)))

    def destroy(self):
        self._cap.release()
        cv2.destroyAllWindows()
        
    def getAwayColors(self):
        if self._awayColors is None:
            self._awayColors = AWAY_TEAM_COLOR_DICT[self._awayTeam]
        return self._awayColors

    def getHomeColors(self):
        if self._homeColors is None:
            self._homeColors = HOME_TEAM_COLOR_DICT[self._homeTeam]
        return self._homeColors

    def getPanningTrajectory(self):
        if self._panningTrajectory is None:
            if os.path.isfile(self._panningTrajectoriesPath):
                self._panningTrajectory = pickle.load(open(self._panningTrajectoriesPath, "rb")).fillna(0)
                if self._verbose:
                    print "Opening trajectories pickle file from {}".format(self._panningTrajectoriesPath)
            else:
                panner = Panner(self._video)
                self._panningTrajectory = panner.getPanningTrajectory().fillna(0)
                pickle.dump(self._panningTrajectory, open(self._panningTrajectoriesPath, "wb"))
                if self._verbose:
                    print "Saving panning trajectory pickle file to {}".format(self._panningTrajectoriesPath)
        return self._panningTrajectory

    def showPoints(self):
        frameObjects = self.getFrameObjects()

        
        savedVidPath = os.path.join(vidDir, '{}-pts.avi'.format(self._videoTitle))
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (self._width, self._height))

        if self._verbose:
            print "Saving points"
            print "Saving to {}".format(savedVidPath)

        for frameObject in frameObjects:
            img = frameObject.drawPoints()
            out.write(img)
            cv2.imshow('points', img)
            cv2.waitKey(20)
        out.release()


    def showHomeCentroids(self):
        frameObjects = self.getFrameObjects()

        savedVidPath = os.path.join(vidDir, '{}-homeCentroids.avi'.format(self._videoTitle))
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (self._width, self._height))

        if self._verbose:
            print "Showing home centroids"
            print "Saving to {}".format(savedVidPath)

        for frameObject in frameObjects:
            img = frameObject.drawHomeMaskCentroids()
            out.write(img)
            cv2.imshow('home centroids', img)
            cv2.waitKey(20)
        out.release()
        

    def showAwayCentroids(self):
        frameObjects = self.getFrameObjects()

        savedVidPath = os.path.join(vidDir, '{}-awayCentroids.avi'.format(self._videoTitle))
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (self._width, self._height))

        if self._verbose:
            print "Showing away centroids"
            print "Saving to {}".format(savedVidPath)

        for frameObject in frameObjects:
            img = frameObject.drawAwayMaskCentroids()
            out.write(img)
            cv2.imshow('away centroids', img)
            cv2.waitKey(20)
        out.release()

    def showSmoothedAwayCentroids(self):
        self.smoothAwayCentroids()
        frameObjects = self.getFrameObjects()

        savedVidPath = os.path.join(vidDir, '{}-smoothedAwayCentroids.avi'.format(self._videoTitle))
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (self._width, self._height))

        if self._verbose:
            print "Showing smoothed away centroids"
            print "Saving to {}".format(savedVidPath)

        for frameObject in frameObjects:
            img = frameObject.drawAwayMaskCentroids()
            out.write(img)
            cv2.imshow('smoothed away centroids', img)
            cv2.waitKey(20)
        out.release()

    def showSmoothedHomeCentroids(self):
        self.smoothHomeCentroids()
        frameObjects = self.getFrameObjects()

        savedVidPath = os.path.join(vidDir, '{}-smoothedHomeCentroids.avi'.format(self._videoTitle))
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (self._width, self._height))

        if self._verbose:
            print "Showing smoothed home centroids"
            print "Saving to {}".format(savedVidPath)

        for frameObject in frameObjects:
            img = frameObject.drawHomeMaskCentroids()
            out.write(img)
            cv2.imshow('smoothed home centroids', img)
            cv2.waitKey(20)
        out.release()

    def smoothHomeCentroids(self):
        if self._verbose:
            print "Smoothing home centroids"
        frameObjects = self.getFrameObjects()
        avgVelocities = self.getAverageVelocities()

        print avgVelocities

        # find first instance where two consecutive frames have at least 5 home players
        #firstValidIdx = 87 # gswokc-5
        #firstValidIdx = 19 # gswokc-1
        #firstValidIdx = 28 # gswokc-4
        #firstValidIdx = 0 # housas-1
        #firstValidIdx = 87 # gswokc-6
        firstValidIdx = 40 # gswokc-8
        lastFrameObject = frameObjects[firstValidIdx]
        for i, frameObject in enumerate(frameObjects[firstValidIdx + 1:]):
            if len(frameObject.getHomeMaskCentroids()) >= 5 and len(lastFrameObject.getHomeMaskCentroids()) >= 5:
                firstValidIdx = i + firstValidIdx
                lastFrameObject = frameObject
                break
            else:
                lastFrameObject = frameObject

        if self._verbose:
            print "First valid index: {}".format(firstValidIdx)
            print "Filling in the home player points backwards until the first index"

        nextFrameObject = frameObjects[firstValidIdx]

        for frameObject, avgVelocity in reversed(list(zip(frameObjects[:firstValidIdx], avgVelocities[:firstValidIdx]))):
            avgVelocity = (-avgVelocity[0], -avgVelocity[1])
            newPts = updatedPlayerPoints(nextFrameObject.getHomeMaskCentroids(),
                                                         frameObject.getHomeMaskCentroids(),
                                                         avgVelocity)
            frameObject.setHomeMaskCentroids(newPts)
            nextFrameObject = frameObject

        for frameObject, avgVelocity in zip(frameObjects[firstValidIdx:], avgVelocities[firstValidIdx:]):
            newPts = updatedPlayerPoints(lastFrameObject.getHomeMaskCentroids(),
                                                         frameObject.getHomeMaskCentroids(),
                                                         avgVelocity)
            frameObject.setHomeMaskCentroids(newPts)
            lastFrameObject = frameObject



    def smoothAwayCentroids(self):
        if self._verbose:
            print "Smoothing away centroids"
        frameObjects = self.getFrameObjects()
        avgVelocities = self.getAverageVelocities()

        print avgVelocities

        # find first instance where two consecutive frames have at least 5 away players
        #firstValidIdx = 31 # gswokc-5
        #firstValidIdx = 18 # gswokc-1
        #firstValidIdx = 164 # gswokc-4
        #firstValidIdx = 0 #housas-1
        #firstValidIdx = 31 # gswokc-6
        firstValidIdx = 32 # gswokc-8
        lastFrameObject = frameObjects[firstValidIdx]
        for i, frameObject in enumerate(frameObjects[firstValidIdx + 1:]):
            if len(frameObject.getAwayMaskCentroids()) >= 5 and len(lastFrameObject.getAwayMaskCentroids()) >= 5:
                firstValidIdx = i + firstValidIdx
                lastFrameObject = frameObject
                break
            else:
                lastFrameObject = frameObject

        if self._verbose:
            print "First valid index: {}".format(firstValidIdx)
            print "Filling in the away player points backwards until the first index"

        nextFrameObject = frameObjects[firstValidIdx]

        for frameObject, avgVelocity in reversed(list(zip(frameObjects[:firstValidIdx], avgVelocities[:firstValidIdx]))):
            avgVelocity = (-avgVelocity[0], -avgVelocity[1])
            newPts = updatedPlayerPoints(nextFrameObject.getAwayMaskCentroids(),
                                                         frameObject.getAwayMaskCentroids(),
                                                         avgVelocity)
            frameObject.setAwayMaskCentroids(newPts)
            nextFrameObject = frameObject

        for frameObject, avgVelocity in zip(frameObjects[firstValidIdx:], avgVelocities[firstValidIdx:]):
            newPts = updatedPlayerPoints(lastFrameObject.getAwayMaskCentroids(),
                                                         frameObject.getAwayMaskCentroids(),
                                                         avgVelocity)
            frameObject.setAwayMaskCentroids(newPts)
            lastFrameObject = frameObject

    def fillQuadranglePoints(self):
        if self._verbose:
            print "Filling quadrangle points with velocity estimates"
        frameObjects = self.getFrameObjects()

        # find the first instance where two consecutive frames have all points
        #firstValidIdx = 0
        firstValidIdx = 38 # gswokc-8
        lastFrameObject = frameObjects[firstValidIdx]
        for i, frameObject in enumerate(frameObjects[firstValidIdx + 1:]):
            if frameObject.allLinesDetected() and lastFrameObject.allLinesDetected():
                firstValidIdx = i + firstValidIdx
                break
            else:
                lastFrameObject = frameObject
        if self._verbose:
            print "First valid index: {}".format(firstValidIdx)


        # Reversing
        if self._verbose:
            print "Filling in the court points backwards until the first index"
        nextFrameObject = frameObjects[firstValidIdx]

        self._avgVelocities = []

        for i, frameObject in reversed(list(enumerate(frameObjects[:firstValidIdx]))):
            avgVelocity, newPts = updatedCourtPoints(nextFrameObject.getQuadranglePoints(), frameObject.getQuadranglePoints())
            frameObject.setQuadranglePts(newPts)
            nextFrameObject = frameObject
            self._avgVelocities.append(avgVelocity)
    
        self._avgVelocities.reverse()

        if self._verbose:
            print "Filling in the court point forwards now"

        for i, frameObject in enumerate(frameObjects[firstValidIdx:]):
            avgVelocity, newPts = updatedCourtPoints(lastFrameObject.getQuadranglePoints(), frameObject.getQuadranglePoints())
            frameObject.setQuadranglePts(newPts)
            lastFrameObject = frameObject
            self._avgVelocities.append(avgVelocity)
        pickle.dump(self._avgVelocities, open(self._avgVelocitiesPath, "wb"))
        if self._verbose:
            print "Saving average velocities pickle file to {}".format(self._avgVelocitiesPath)

    def getAverageVelocities(self):
        if self._avgVelocities is None:
            if os.path.isfile(self._avgVelocitiesPath):
                self._avgVelocities = pickle.load(open(self._avgVelocitiesPath, "rb"))
                if self._verbose:
                    print "Opening average velocities pickle file from {}".format(self._avgVelocitiesPath)
            else:
                self.fillQuadranglePoints()
        return self._avgVelocities

    def fillQuadranglePointsWithTrajectories(self):
        # Fills in the quadrangle points for all of the frame objects in case it doesn't exist already
        # It uses the panning trajectories to do this
        if self._verbose:
            print "Setting quadrangle points"
        frameObjects = self.getFrameObjects()
        trajectories = self.getPanningTrajectory()
        # Get all possible quadrangle points 
        allPts = []
        for frameObject in frameObjects:
            pts = frameObject.getQuadranglePoints()
            allPts.append(pts)

        # Note - index of trajectory i is the trajectory from frame i to i+1
        # e.g. trajectory[0]  is the trajectory from 0 to 1`
        
        # Algorithm:
        # 1. Find the first valid points in the list
        # 2. Calculate the new quadrangle coordinates with a transform 
        #    of the last known point (going in reverse)
        # 3. Set the updated points in the frame
        # 4. Repeat steps 2 and 3 except forward until there are no more nones in the list
        firstValidIdx = notNoneIdx(allPts)

        if self._verbose:
            print "First valid index was: {}".format(firstValidIdx)
        
        lastFrameObject = frameObjects[firstValidIdx]
        lastPoints = lastFrameObject.getQuadranglePoints()

        if self._verbose:
            print "Updating points backwards..."

        # Go in reverse and adjust the locations using panning
        for i, frameObject in reversed(list(enumerate(frameObjects[:firstValidIdx]))):
            trajectory = trajectories.loc[i]
            updatedCourtPoints = [[component - dcomponent for component,dcomponent in zip(pt, trajectory)]  for pt in lastPoints] 
            frameObject.setQuadranglePts(updatedCourtPoints)
            allPts[i] = frameObject.getQuadranglePoints()

        # Go forward and update based on trajectory
        lastPoints = frameObject.getQuadranglePoints()
        for i, frameObject in enumerate(frameObjects[firstValidIdx:]):
            if allPts[i] is None or frameObject.getQuadranglePoints() is None:
                trajectory = trajectories.loc[i]
                updatedCourtPoints = [[component + dcomponent for component,dcomponent in zip(pt, trajectory)]  for pt in lastPoints] 
                frameObject.setQuadranglePts(updatedCourtPoints)
                allPts[i] = frameObject.getQuadranglePoints()
            else:
                lastPoints = frameObject.getQuadranglePoints()
    
    def showFilledPoints(self):
        self.fillQuadranglePoints()

        savedVidPath = os.path.join(vidDir, '{}-filledPts.avi'.format(self._videoTitle))
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (self._width, self._height))
        if self._verbose:
            print "Showing filled quadrangle points"
            print "Saving to {}".format(savedVidPath)

        for frameObject in self.getFrameObjects():
            img = frameObject.drawPoints()
            out.write(img)
            cv2.imshow('filled points', img)
            cv2.waitKey(20)
        out.release()

    def getAwayPlayerPoints(self):
        if self._awayPlayerPositions is None:
            self._awayPlayerPositions = []
            for frameObject in self.getFrameObjects():
                self._awayPlayerPositions.append(frameObject.getAwayMaskCentroids()) 
        return self._awayPlayerPositions

    def getHomePlayerPoints(self):
        if self._homePlayerPositions is None:
            self._homePlayerPositions = []
            for frameObject in self.getFrameObjects():
                self._homePlayerPositions.append(frameObject.getHomeMaskCentroids()) 
        return self._homePlayerPositions

    def getHomographies(self):
        if self._homographies is None:
            self._homographies = []
            frameObjects = self.getFrameObjects()
            for frameObject in frameObjects:
                self._homographies.append(frameObject.getHomography())
        return self._homographies


    def getCourtPoints(self):
        if self._courtPoints is None:
            self._courtPoints = []
            frameObjects = self.getFrameObjects()
            for frameObject in frameObjects:
                self._courtPoints.append(frameObject.getQuadranglePoints())
        return self._courtPoints

    def getAwayTrajectories(self):
        if self._awayTrajectories is None:
            frameObjects = self.getFrameObjects()
        return self._awayTrajectories

    def getHomeTrajectories(self):
        if self._homeTrajectories is None:
            frameObjects = self.getFrameObjects()
        return self._homeTrajectories

    def getFrameObjects(self):
        if self._frameObjects is None:
            if os.path.isfile(self._frameObjectsPath):
                self._frameObjects = pickle.load(open(self._frameObjectsPath, "rb"))
                assert len(self._frameObjects) == self._capLength - 1 # Sanity check
                if self._verbose:
                    print "Opening frame object pickle file from {}".format(self._frameObjectsPath)
            else:
                if self._verbose:
                    print "No fo pkl file found from {}. Generating frame objects".format(self._frameObjectsPath)
                self._frameObjects = []
                frameNum = 0
                while(frameNum < self._capLength - 1):
                    ret, img = self._cap.read()
                    frameNum += 1

                    if self._verbose:
                        print "Processing frame {}/{}".format(frameNum, self._capLength)
                    try:
                        frameObject = fo.FrameObject(img, frameNum, self._video, 
                                self.getAwayColors(), self.getHomeColors(), self._side, self._verbose)
                        self._frameObjects.append(frameObject)
                        if self._verbose:
                            print "Generating"
                        frameObject.generate()
                    except Exception as e:
                        print (e)
                pickle.dump(self._frameObjects, open(self._frameObjectsPath, "wb"))
                if self._verbose:
                    print "Saving fo pickle file to {}".format(self._frameObjectsPath)
        return self._frameObjects


    def showAdjustedPoints(self):
        if self._verbose:
            print "Adjusting the points with homography"
        frameObjects = self.getFrameObjects()
        self.fillQuadranglePoints()
        self.smoothHomeCentroids()
        self.smoothAwayCentroids()

        canvasWidth = nba.getWidth()
        canvasHeight = nba.getHeight()

        # To use hstack we need to make the images the same dimensions
        # Therefore we take the bigger of the two images and pad the other

        adjustedVideoHeight = max(canvasHeight, self._height)
        joinedWidth = canvasWidth / 2 + self._width
        joinedHeight = adjustedVideoHeight


        scale_x = 0.75
        scale_y = 0.75

        savedVidPath = os.path.join(vidDir, '{}-adjusted.avi'.format(self._videoTitle))
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (int(joinedWidth*scale_x), int(joinedHeight*scale_y)))

        allHomographies = self.getHomographies()
        allAwayPoints = self.getAwayPlayerPoints()
        allHomePoints = self.getHomePlayerPoints()
        allCourtPoints = self.getCourtPoints()


        allAdjustedHomePoints = []
        allAdjustedAwayPoints = []
        allAdjustedCourtPoints = []

        awayCircleColor = frameObjects[0].getAwayCircleColor()
        homeCircleColor = frameObjects[0].getHomeCircleColor()

        x_translation = 200 if self._side == 'left' else -200
        y_translation = 200

        for homographies, awayPoints, homePoints, courtPoints in zip(allHomographies, allAwayPoints, allHomePoints, allCourtPoints):
            awayPoints = np.array([awayPoints], dtype='float32')
            homePoints = np.array([homePoints], dtype='float32')
            courtPoints = np.array([courtPoints], dtype='float32')

            adjustedAwayPoints = cv2.perspectiveTransform(awayPoints, homographies)
            adjustedHomePoints = cv2.perspectiveTransform(homePoints, homographies)
            adjustedCourtPoints = cv2.perspectiveTransform(courtPoints, homographies)


            allAdjustedHomePoints.append(adjustedHomePoints)
            allAdjustedAwayPoints.append(adjustedAwayPoints)
            allAdjustedCourtPoints.append(adjustedCourtPoints)

            if self._verbose:
                print "away (before): {}".format(awayPoints)
                print "home (before): {}".format(homePoints)
                print "away (after): {}".format(adjustedAwayPoints)
                print "home (after): {}".format(adjustedHomePoints)

        #blank = np.zeros([canvasWidth, canvasHeight, 3], np.uint8)
        blank = np.zeros([canvasHeight, canvasWidth, 3], np.uint8)

        assert len(allAdjustedHomePoints) == len(allAdjustedAwayPoints)


        for adjustedHomePts, adjustedAwayPts, frameObject, courtPts in zip(allAdjustedHomePoints, allAdjustedAwayPoints, frameObjects, allAdjustedCourtPoints):
            assert len(adjustedHomePts) == len(adjustedAwayPts)

            canvas = blank.copy()
            for (x_away, y_away), (x_home, y_home) in zip(adjustedAwayPts[0], adjustedHomePts[0]):
                circs = cv2.circle(canvas, (int(x_away + x_translation), int(y_away + y_translation)), 5, homeCircleColor, -1)
                circs = cv2.circle(canvas, (int(x_home + x_translation), int(y_home + y_translation)), 5, awayCircleColor, -1)
            for (x, y) in courtPts[0]:
                circs = cv2.circle(canvas, (int(x + x_translation), int(y + y_translation)), 5, colors.BGR_RED)

            origImg = frameObject.drawPoints()
            origImg = frameObject.drawAwayMaskCentroids(origImg)
            origImg = frameObject.drawHomeMaskCentroids(origImg)

            # Combine the images
            joined = np.zeros([joinedHeight, joinedWidth, 3], np.uint8)

            joined[:self._height, :self._width] = origImg
            if self._side == 'left':
                joined[:canvasHeight, self._width:self._width + canvasWidth / 2] = canvas[:,:canvasWidth / 2]
            else:
                joined[:canvasHeight, self._width:self._width + canvasWidth / 2] = canvas[:,canvasWidth / 2:]
            joined = cv2.resize(joined, (0,0), fx = scale_x, fy = scale_y)
            cv2.imshow('adjusted', joined)
            out.write(joined)
            cv2.waitKey(20)
        out.release()


    def show(self):
        self.fillQuadranglePoints()
        frameObjects = self.getFrameObjects()
        for frameObject in frameObjects:
            frameObject.show()
            cv2.waitKey(20)


def testEngineMain(args):
    engine = Engine(args.vid, args.away, args.home, args.side, args.verbose)
    if args.mode == 'show':
        if args.awayPoints:
            engine.showAwayCentroids()
        if args.homePoints:
            engine.showHomeCentroids()
        if args.court:
            engine.showPoints()
        if args.courtFilled:
            engine.showFilledPoints()
        if args.smoothedAwayPoints:
            engine.showSmoothedAwayCentroids()
        if args.smoothedHomePoints:
            engine.showSmoothedHomeCentroids()
        if args.adjusted:
            engine.showAdjustedPoints()
    engine.destroy()

def main(args):
    cap = cv2.VideoCapture(args.vid)

    currentFrame = 0
    capLength = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    if args.verbose:
        print(("#===== Opening {} =====#".format(args.vid)))
        print(("Successful?: {}".format(str(cap.isOpened()))))
        print(("Frame width: {}".format(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))))
        print(("Frame height: {}".format(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))))
        print(("FPS: {}".format(str(cap.get(cv2.cv.CV_CAP_PROP_FPS)))))
        print(("Frame count: {}".format(capLength)))

    while(currentFrame < capLength):
        ret, img = cap.read()

        if args.verbose:
            print(("Processing frame {}/{}".format(currentFrame, capLength)))

        # Operations goes here

        try:
            currentFrame += 1
            """
            if currentFrame < 45:
                continue
            """
            frameObject = fo.FrameObject(img, currentFrame, args.vid, colors.GSW_AWAY, colors.OKC_HOME)
            #frameObject = fo.FrameObject(img, currentFrame, args.vid, colors.HOU_AWAY, colors.SAS_HOME)

            #frameObject.showAwayJerseyMask()
            #frameObject.showAwayMaskCentroids()
            #frameObject.showHomeJerseyMask()
            #frameObject.showHomeMaskCentroids()
            frameObject.show()

            #frameObject.show_home_player_coordinates()

            #frameObject.show_lines()
            #frameObject.show_points()
            #raw_input()


        except Exception as inst:
            print(inst)
            print "Continuing"

        # Display things here
        #cv2.imshow('frame', img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--verbose', action='store_true',
                        default=False)
    
    parser.add_argument('--vid', type=str,
                        help="Video file.",
                        default=defaultVidPath)

    parser.add_argument('--away', type=str,
                        help="Away team",
                        #default='HOU')
                        default='GSW')


    parser.add_argument('--side', type=str,
                        help="Side of the court",
                        #default='right')
                        default='left')

    parser.add_argument('--home', type=str,
                        help="Home team",
                        #default='SAS')
                        default='OKC')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    showParser = subparsers.add_parser(
        'show', help='Show things related to the court.')

    showParser.add_argument('--awayPoints', action='store_true',
                             help='Show away mask centroids')

    showParser.add_argument('--homePoints', action='store_true',
                             help='Show home mask centroids')

    showParser.add_argument('--court', action='store_true',
                             help='Show the court points')

    showParser.add_argument('--courtFilled', action='store_true',
                             help='Show the court points adjusted with panning')

    showParser.add_argument('--smoothedAwayPoints', action='store_true',
                             help='Show away mask centroids')

    showParser.add_argument('--smoothedHomePoints', action='store_true',
                             help='Show home mask centroids')

    showParser.add_argument('--adjusted', action='store_true',
                             help='Show all points adjusted')

    args = parser.parse_args()

    testEngineMain(args)
    #main(args)
