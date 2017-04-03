import errno
import os

import argparse
import numpy as np
import cv2

import colors


fileDir = os.path.dirname(os.path.realpath(__file__))
vidDir = os.path.join(fileDir, '..', 'videos')

defaultVidPath = os.path.join(vidDir, 'gswokc-1.mp4')


def main(args):
    cap = cv2.VideoCapture(args.vid)

    currentFrame = 0
#    capLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    capLength = 1

    if args.verbose:
        print("#===== Opening {} =====#".format(args.vid))
        print("Successful?: {}".format(str(cap.isOpened())))
        print("Frame width: {}".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("Frame height: {}".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("FPS: {}".format(str(cap.get(cv2.CAP_PROP_FPS))))
        print("Frame count: {}".format(capLength))


    while(currentFrame < capLength):
        ret, img = cap.read()

        if args.verbose:
            print ("Processing frame {}/{}".format(currentFrame, capLength))



        if (currentFrame == capLength - 1):
            court_mask = colors.get_court_mask(img)
            cv2.imshow('court_mask', court_mask)
            cv2.imshow('frame', img)


        # Any operation goes here

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #cv2.imshow('frame', img)
        currentFrame += 1

    input("Waiting")
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
