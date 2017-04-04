import errno
import os

import argparse
import numpy as np
import cv2

import frame_object as fo


fileDir = os.path.dirname(os.path.realpath(__file__))
vidDir = os.path.join(fileDir, '..', 'videos')

defaultVidPath = os.path.join(vidDir, 'uclawash-1.mp4')

def main(args):
    cap = cv2.VideoCapture(args.vid)

    currentFrame = 0
#    capLength = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    capLength = 210

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
            frameObject = fo.FrameObject(img, currentFrame, args.vid)
            #frameObject.show_lines()
            frameObject.show_points()
            raw_input()
        except Exception as inst:
            print(inst)
            print "Continuing"

        currentFrame += 1

        # Display things here
        cv2.imshow('frame', img)


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
