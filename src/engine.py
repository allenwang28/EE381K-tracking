import errno
import os

import argparse
import numpy as np
import cv2

import frame_object as fo
import colors


fileDir = os.path.dirname(os.path.realpath(__file__))
vidDir = os.path.join(fileDir, '..', 'videos')

defaultVidPath = os.path.join(vidDir, 'housas-1.mp4')
#defaultVidPath = os.path.join(vidDir, 'gswokc-1.mp4')

# NOTE - this is required because we only found the 
# ranges for a limited number of teams
ACCEPTED_AWAY_TEAM_KEYS = [
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


class Engine:
    # Video/opencv related variables
    _video = None # This is a path
    _cap = None
    _capLength = None

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


    def __init__(self, video, awayTeam, homeTeam, verbose = True):
        self._video = video
        if awayTeam not in ACCEPTED_AWAY_TEAM_KEYS:
            raise Exception("Invalid away team provided: {}".format(awayTeam))
        if homeTeam not in ACCEPTED_HOME_TEAM_KEYS:
            raise Exception("Invalid home team provided: {}".format(homeTeam))
        self._awayTeam = awayTeam
        self._homeTeam = homeTeam
        self._verbose = verbose
        self._cap = cv2.VideoCapture(_video)
        self._capLength = int(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        if verbose:
            print(("#===== Opening {} =====#".format(video)))
            print(("Successful?: {}".format(str(self._cap.isOpened()))))
            print(("Frame width: {}".format(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))))
            print(("Frame height: {}".format(self._cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))))
            print(("FPS: {}".format(str(self._cap.get(cv2.cv.CV_CAP_PROP_FPS)))))
            print(("Frame count: {}".format(self._capLength)))

    def getAwayColors(self):
        if self._awayColors is None:
            self._awayColors = away_TEAM_COLOR_DICT[self._awayTeam]
        return self._awayColors

    def getHomeColors(self):
        if self._homeColors is None:
            self._homeColors = HOME_TEAM_COLOR_DICT[self._homeTeam]
        return self._homeColors

    def getFrameObjects(self):
        # TODO - save all of this into a pickle file and load
        if self._frameObjects is None:
            self._frameObjects = []
            frameNum = 0
            while(frameNum < self._capLength):
                ret, img = self._cap.read()

                if self._verbose:
                    print "Processing frame {}/{}".format(frameNum, self._capLength)
                try:
                    frameObject = fo.FrameObject(img, frameNum, self._video, 
                            self.getAwayColors(), self.getHomeColors())
                    frameObject.generate()
                    self._frameObjects.append(frameObject)
                    if self._verbose:
                        frameObject.show()

                except Exception as e:
                    if self._verbose:
                        print (e)
        return self._frameObjects


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
            #frameObject = fo.FrameObject(img, currentFrame, args.vid, GSW_AWAY, OKC_HOME)
            frameObject = fo.FrameObject(img, currentFrame, args.vid, colors.HOU_AWAY, colors.SAS_HOME)
            #frameObject.show_away_mask_centroids()
            #frameObject.show_away_jersey_mask()
            frameObject.show()

            #frameObject.show_home_player_coordinates()

            #frameObject.show_lines()
            #frameObject.show_points()
            #raw_input()


        except Exception as inst:
            print(inst)
            print "Continuing"
        currentFrame += 1

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

    args = parser.parse_args()

    main(args)
