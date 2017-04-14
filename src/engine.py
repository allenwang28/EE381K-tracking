import errno
import os

import argparse
import numpy as np
import cv2

import frame_object as fo
import colors

import panning
from panning import Panner

import pickle


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



class Engine:
    # Video/opencv related variables
    _video = None # This is a path
    _cap = None
    _capLength = None
    _side = None

    # Lists
    _frameObjects = None
    _trackers = None

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
        videoTitle = os.path.splitext(os.path.basename(video))[0]
        self._frameObjectsPath = os.path.join(dataPath, '{}-frames.p'.format(videoTitle))
        self._panningTrajectoriesPath = os.path.join(dataPath, '{}-traj.p'.format(videoTitle))
        if verbose:
            print(("#===== Opening {} =====#".format(video)))
            print(("Successful?: {}".format(str(self._cap.isOpened()))))
            print(("Frame width: {}".format(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))))
            print(("Frame height: {}".format(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))))
            print(("FPS: {}".format(str(self._cap.get(cv2.cv.CV_CAP_PROP_FPS)))))
            print(("Frame count: {}".format(self._capLength)))

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
                    print "Opening fo pickle file from {}".format(self._frameObjectsPath)
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
                                self.getAwayColors(), self.getHomeColors(), self._side)
                        self._frameObjects.append(frameObject)
                        if self._verbose:
                            print "Generating"
                        frameObject.generate()
                        if self._verbose:
                            frameObject.show()

                    except Exception as e:
                        if self._verbose:
                            print (e)
                pickle.dump(self._frameObjects, open(self._frameObjectsPath, "wb"))
                if self._verbose:
                    print "Saving fo pickle file to {}".format(self._frameObjectsPath)
        return self._frameObjects


def testEngineMain(args):
    engine = Engine(args.vid, args.away, args.home, args.side, args.verbose)
    print engine.getHomographies()

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
                        default=True)
    
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

    args = parser.parse_args()

    testEngineMain(args)
    #main(args)
