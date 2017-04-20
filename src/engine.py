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

import pickle

import math


fileDir = os.path.dirname(os.path.realpath(__file__))
vidDir = os.path.join(fileDir, '..', 'videos')
dataPath = os.path.join(fileDir, '..', 'data')

#defaultVidPath = os.path.join(vidDir, 'housas-1.mp4')
defaultVidPath = os.path.join(vidDir, 'gswokc-5.mp4')

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

def updatedPoints(lastPts, thisPts):
    print "=== # Getting average velocity # ==="
    #assert len(lastPts) == len(thisPts)
    xVelocities = []
    yVelocities = []
    avgVelocity = None

    # First calculate the average velocity
    for lastPt, thisPt in zip(lastPts, thisPts):
        print "lastPt: {}, thisPt: {}".format(lastPt, thisPt)
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
    print "Average velocity: {}".format(avgVelocity)
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
    _homeTrajectories = None
    _awayTrajectories = None

    _frameObjectsPath = None
    _panningTrajectoriesPath = None

    _trackers = None


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
        self._panningTrajectoriesPath = os.path.join(dataPath, '{}-traj.p'.format(self._videoTitle))

        self._fourcc = cv2.cv.CV_FOURCC(*'X264')

        if verbose:
            print(("#===== Opening {} =====#".format(video)))
            print(("Successful?: {}".format(str(self._cap.isOpened()))))
            print(("Frame width: {}".format(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))))
            print(("Frame height: {}".format(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))))
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
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (1280, 720))

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
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (1280, 720))

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
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (1280, 720))

        if self._verbose:
            print "Showing away centroids"
            print "Saving to {}".format(savedVidPath)

        for frameObject in frameObjects:
            img = frameObject.drawAwayMaskCentroids()
            out.write(img)
            cv2.imshow('away centroids', img)
            cv2.waitKey(20)
        out.release()

    # This doesn't work
    def getAwayTrackers(self):
        if self._verbose:
            print "Setting away trackers"
        frameObjects = self.getFrameObjects()

        firstValidIdx = 0
        centroids = None
        # Look for the first frame object with only 5 jerseys detected 
        for i, frameObject in enumerate(frameObjects):
            if len(frameObject.getAwayMaskCentroids()) == 5:
                firstValidIdx = i 
                centroids = frameObject.getAwayMaskCentroids()
        if centroids is None:
            raise Exception("No valid away centroids detected")
        self._awayTrackers = []

        for centroid in centroids:
            self._awayTrackers.append(Tracker(frameObjects[firstValidIdx].getBgrImg(),
                                              centroid[0],
                                              5,
                                              centroid[1],
                                              5))

        for frameObject in frameObjects[firstValidIdx:]:
            frame = frameObject.getBgrImg()
            for tracker in self._awayTrackers:
                if tracker.isLive():
                    tracker.drawOnFrame(frame)
            cv2.imshow('away trajectory', frame)
            cv2.waitKey(20)

    def fillQuadranglePoints(self):
        self.testFunction()
        if self._verbose:
            print "Filling quadrangle points with velocity estimates"
        frameObjects = self.getFrameObjects()

        # find the first instance where two consecutive frames have all points
        lastFrameObject = frameObjects[0]
        firstValidIdx = 0
        for i, frameObject in enumerate(frameObjects[1:]):
            if frameObject.allLinesDetected() and lastFrameObject.allLinesDetected():
                firstValidIdx = i
                break
            else:
                lastFrameObject = frameObject
        if self._verbose:
            print "First valid index: {}".format(firstValidIdx)

        for i, frameObject in enumerate(frameObjects[firstValidIdx:]):
            newPts = updatedPoints(lastFrameObject.getQuadranglePoints(), frameObject.getQuadranglePoints())

            """
            avgVelocity = getAverageVelocity(lastFrameObject, frameObject)
            print "avgVelocity = {}".format(avgVelocity)
            newPts = []
            newPts = [(oldPt[0] + avgVelocity[0],oldPt[1] + avgVelocity[1]) for oldPt in lastFrameObject.getQuadranglePoints()]
            frameObject.setQuadranglePts(newPts)
            # TODO - do something if the average velocity is over some threshold
            """
            print "New points: {}".format(newPts)
            frameObject.setQuadranglePts(newPts)
            lastFrameObject = frameObject

    def testFunction(self):
        if self._verbose:
            print "Finding first index for glitch"
        frameObjects = self.getFrameObjects()
        idx = 0
        for i, frameObject in enumerate(frameObjects):
            if len(frameObject.getQuadranglePoints()) != 4:
                idx = i
        print idx

        raw_input()


    
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
            updatedPoints = [[component - dcomponent for component,dcomponent in zip(pt, trajectory)]  for pt in lastPoints] 
            frameObject.setQuadranglePts(updatedPoints)
            allPts[i] = frameObject.getQuadranglePoints()

        # Go forward and update based on trajectory
        lastPoints = frameObject.getQuadranglePoints()
        for i, frameObject in enumerate(frameObjects[firstValidIdx:]):
            if allPts[i] is None or frameObject.getQuadranglePoints() is None:
                trajectory = trajectories.loc[i]
                updatedPoints = [[component + dcomponent for component,dcomponent in zip(pt, trajectory)]  for pt in lastPoints] 
                frameObject.setQuadranglePts(updatedPoints)
                allPts[i] = frameObject.getQuadranglePoints()
            else:
                lastPoints = frameObject.getQuadranglePoints()
    
    def showFilledPoints(self):
        self.fillQuadranglePoints()

        savedVidPath = os.path.join(vidDir, '{}-filledPts.avi'.format(self._videoTitle))
        out = cv2.VideoWriter(savedVidPath, -1, 20.0, (1280, 720))
        if self._verbose:
            print "Showing filled quadrangle points"
            print "Saving to {}".format(savedVidPath)

        for frameObject in self.getFrameObjects():
            img = frameObject.drawPoints()
            out.write(img)
            cv2.imshow('filled points', img)
            cv2.waitKey(20)
        out.release()

    def getHomographies(self):
        if self._homographies is None:
            if self._verbose:
                print "Calculating homographies"
            frameObjects = self.getFrameObjects()
            trajectories = self.getPanningTrajectory()
            # Get all possible homographies
            homographies = []
            for frameObject in frameObjects:
                h = frameObject.getHomography()
                homographies.append(h)
            # Fill in the nones with previous homographies adjusted by the panning

            # Note - index of trajectory i is the trajectory from frame i to i+1
            # e.g. trajectory[0]  is the trajectory from 0 to 1`
            
            # Algorithm:
            # 1. Find the first valid homography in the list
            # 2. Calculate the new quadrangle coordinates with a transform 
            #    of the last known homography (going in reverse)
            # 3. Set the updated points in the frame, calculate homography 
            # 4. Repeat steps 2 and 3 except forward until there are no more nones in the list
            firstValidIdx = notNoneIdx(homographies)

            if self._verbose:
                print "First valid index was: {}".format(firstValidIdx)
            
            lastFrameObject = frameObjects[firstValidIdx]
            lastPoints = lastFrameObject.getQuadranglePoints()

            if self._verbose:
                print "Updating homographies backwards..."

            # Go in reverse and adjust the locations using panning
            for i, frameObject in reversed(list(enumerate(frameObjects[:firstValidIdx]))):
                trajectory = trajectories.loc[i]
                updatedPoints = [[component - dcomponent for component,dcomponent in zip(pt, trajectory)]  for pt in lastPoints] 
                frameObject.setQuadranglePts(updatedPoints)
                homographies[i] = frameObject.getHomography()

            # Go forward and update based on trajectory
            lastPoints = frameObject.getQuadranglePoints()
            for i, frameObject in enumerate(frameObjects[firstValidIdx:]):
                if homographies[i] is None or frameObject.getQuadranglePoints() is None:
                    trajectory = trajectories.loc[i]
                    updatedPoints = [[component + dcomponent for component,dcomponent in zip(pt, trajectory)]  for pt in lastPoints] 
                    frameObject.setQuadranglePts(updatedPoints)
                    homographies[i] = frameObject.getHomography()
                else:
                    lastPoints = frameObject.getQuadranglePoints()
            self._homographies = homographies 
            print homographies
        return self._homographies


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
                        if self._verbose:
                            frameObject.show()

                    except Exception as e:
                        print (e)
                pickle.dump(self._frameObjects, open(self._frameObjectsPath, "wb"))
                if self._verbose:
                    print "Saving fo pickle file to {}".format(self._frameObjectsPath)
        return self._frameObjects

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
                        default='GSW')


    parser.add_argument('--side', type=str,
                        help="Side of the court",
                        default='left')

    parser.add_argument('--home', type=str,
                        help="Home team",
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

    args = parser.parse_args()

    testEngineMain(args)
    #main(args)
