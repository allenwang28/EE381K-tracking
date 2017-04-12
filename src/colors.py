# Imports ---------------------------------------

import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque


# Constants -------------------------------------

GSW_AWAY_LOWER = (110, 190, 80)
GSW_AWAY_UPPER = (125, 255, 150)
GSW_AWAY = (GSW_AWAY_LOWER, GSW_AWAY_UPPER)

OKC_HOME_LOWER = (125, 25, 178)
OKC_HOME_UPPER = (180, 75, 255)
OKC_HOME = (OKC_HOME_LOWER, OKC_HOME_UPPER)

HOU_AWAY_LOWER = (170, 85, 100)
HOU_AWAY_UPPER = (180, 255, 255)
HOU_AWAY = (HOU_AWAY_LOWER, HOU_AWAY_UPPER)

SAS_HOME_LOWER = (120, 10, 168)
SAS_HOME_UPPER = (190, 30, 260)
SAS_HOME = (SAS_HOME_LOWER, SAS_HOME_UPPER)


#CROWD_TOP_HEIGHT_FRACTION = .375;
CROWD_TOP_HEIGHT_FRACTION = .31;
CROWD_BOTTOM_HEIGHT_FRACTION = .2;


MID_TOP_HEIGHT_FRACTION = .5;
MID_BOTTOM_HEIGHT_FRACTION = .4;

BGR_BLACK = (0,0,0)
BGR_RED = (0, 0, 255)
BGR_BLUE = (255, 0, 0)
YCBCR_BLACK = (0,128,128)
YCBCR_WHITE = (255,128,128)


# Exported code ---------------------------------
def remap_from_crowdless_coords(original_img, coords):
    new_coords = []
    for (x,y) in coords:
        new_coords.append((x, int(y + CROWD_TOP_HEIGHT_FRACTION*original_img.shape[0])))
    return new_coords 

def get_crowdless_image(img):
    return img[int(CROWD_TOP_HEIGHT_FRACTION*img.shape[0]) : int(-CROWD_BOTTOM_HEIGHT_FRACTION*img.shape[0])]

# NOTE - doesn't work
def get_middle_court(img):
    return img[int(MID_TOP_HEIGHT_FRACTION*img.shape[0]) : int(-MID_BOTTOM_HEIGHT_FRACTION*img.shape[0])]

# NOTE - doesn't work
def get_jersey1_colors(_bgr_img, thresh=0.02, peak_num=2):
    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2YCR_CB)

    get_middle_court(_bgr_img)

    hist = cv2.calcHist([img], [1,2], None, [256,256], [0,256, 0,256])

    subtracted_hist = hist.copy()
    connected_hist = None

    peak_flat_idx = np.argmax(subtracted_hist)
    peak_idx = np.unravel_index(peak_flat_idx, subtracted_hist.shape)
    peak_val = hist[peak_idx]
    connected_hist, sumX, subtracted_hist = get_connected_hist(subtracted_hist, peak_idx, 0.02)


    for _ in xrange(peak_num):
        peak_flat_idx = np.argmax(subtracted_hist)
        peak_idx = np.unravel_index(peak_flat_idx, subtracted_hist.shape)
        peak_val = hist[peak_idx]
        connected_hist, sumX, subtracted_hist = get_connected_hist(subtracted_hist, peak_idx, thresh)

    return connected_hist

def get_jersey_mask(_bgr_img, lower, upper):
    img_hsv = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.inRange(img_hsv, lower, upper)


    # Morphology
    element = np.ones((5,5)).astype(np.uint8)
    img_hsv = cv2.erode(img_hsv, element)
    img_hsv = cv2.dilate(img_hsv, element)
    return img_hsv

def create_court_mask(_bgr_img, dominant_colorset, binary_gray=False):
    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2YCR_CB)
    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            idx = (row, col)
            _, cr, cb = img[idx]
            if (cr, cb) not in dominant_colorset:
                img[idx] = YCBCR_BLACK
            elif binary_gray:
                img[idx] = YCBCR_WHITE

    return ycbcr_to_gray(img) if binary_gray else img

def get_color_hist(_bgr_img, thresh=0.02, ignore_crowd=True, peak_num=1):
    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2YCR_CB)

    if ignore_crowd:
        get_crowdless_image(_bgr_img)

    hist = cv2.calcHist([img], [1,2], None, [256,256], [0,256, 0,256])
    # return hist

    peak1_flat_idx = np.argmax(hist)
    peak1_idx = np.unravel_index(peak1_flat_idx, hist.shape)
    peak1_val = hist[peak1_idx]
    connected_hist1, sum1, subtracted_hist = get_connected_hist(hist, peak1_idx, thresh)
    return connected_hist1
    # return get_connected_hist(hist, peak1_idx, thresh)


def get_dominant_colorset(_bgr_img, thresh=0.02, ignore_crowd=True,
    peak_num=1):
    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2YCR_CB)

    if ignore_crowd:
        get_crowdless_image(_bgr_img)

    hist = cv2.calcHist([img], [1,2], None, [256,256], [0,256, 0,256])

    subtracted_hist = hist.copy()
    connected_hist = None

    for _ in xrange(peak_num):
        peak_flat_idx = np.argmax(subtracted_hist)
        peak_idx = np.unravel_index(peak_flat_idx, subtracted_hist.shape)
        peak_val = hist[peak_idx]
        connected_hist, sumX, subtracted_hist = get_connected_hist(subtracted_hist, peak_idx, thresh)
    return connected_hist

"""
#.02
def get_dominant_colorset(_bgr_img, thresh=0.02, ignore_crowd=True,
    peak_num=1):
    img = cv2.cvtColor(_bgr_img, cv2.COLOR_BGR2YCR_CB)

    if ignore_crowd:
        get_crowdless_image(_bgr_img)

    hist = cv2.calcHist([img], [1,2], None, [256,256], [0,256, 0,256])

    peak1_flat_idx = np.argmax(hist)
    peak1_idx = np.unravel_index(peak1_flat_idx, hist.shape)
    peak1_val = hist[peak1_idx]
    connected_hist1, sum1, subtracted_hist = get_connected_hist(hist, peak1_idx, thresh)

    if peak_num == 1:
        return connected_hist1

    peak2_flat_idx = np.argmax(subtracted_hist)
    peak2_idx = np.unravel_index(peak2_flat_idx, subtracted_hist.shape)
    peak2_val = hist[peak2_idx]
    connected_hist2, sum2, subtracted_hist = get_connected_hist(subtracted_hist, peak2_idx, thresh)

    return connected_hist2
"""

def get_double_flooded_mask(gray_mask):
    gray_flooded = fill_holes_with_contour_filling(gray_mask)
    gray_flooded2 = fill_holes_with_contour_filling(gray_flooded, inverse=True)
    return gray_flooded2

def fill_holes_with_contour_filling(gray_mask, inverse=False):
  filled = gray_mask.copy()
  if inverse:
    filled = cv2.bitwise_not(filled)
  contour, _ = cv2.findContours(filled,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contour:
    cv2.drawContours(filled, [cnt], 0, 255, -1)
  if inverse:
    filled = cv2.bitwise_not(filled)
  return filled


# Non-exported code -----------------------------

def get_connected_hist(hist, peak_idx, thresh):
    connected_hist = set()
    sum_val = 0
    subtracted_hist = np.copy(hist)

    min_passing_val = thresh * hist[peak_idx]

    connected_hist.add(peak_idx)
    sum_val += hist[peak_idx]
    subtracted_hist[peak_idx] = 0
    queue = deque([peak_idx])
    while queue:
        x, y = queue.popleft()
        toAdd = []
        if x > 1:
            toAdd.append((x-1, y))
        if x < hist.shape[0] - 1:
            toAdd.append((x+1, y))
        if y > 1:
            toAdd.append((x, y-1))
        if y < hist.shape[1] - 1:
            toAdd.append((x, y+1))

        for idx in toAdd:
            if idx not in connected_hist and hist[idx] >= min_passing_val:
                connected_hist.add(idx)
                sum_val += hist[idx]
                subtracted_hist[idx] = 0
                queue.append(idx)

    return connected_hist, sum_val, subtracted_hist


# Helpers ---------------------------------------

def show_image(img):
    cv2.imshow('Showing image',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def show_hist(img, hist):
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()

def show_hist_list(hist_list):
    for i, hist in enumerate(hist_list):
        plt.subplot(1, len(hist_list), i+1)
        plt.imshow(hist, interpolation = 'nearest')
    plt.show()


def ycbcr_to_bgr(ycbcr_img):
    img = ycbcr_img.copy()
    return cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)


def ycbcr_to_gray(ycbcr_img):
    img = ycbcr_img.copy()
    img = ycbcr_to_bgr(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ycbcr_to_binary(ycbcr_img):
    img = ycbcr_img.copy()
    return ycbcr_to_gray(img) > 128

def binary_to_gray(binary_img):
    img = binary_img.copy()
    return img * 255;

def gray_to_bgr(gray_img):
    img = gray_img.copy()
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def show_binary(binary):
    plt.imshow(binary)
    plt.show()

# hsv is a set
# This is a hacky way to turn hsv set into an bgr set
def hsv_to_bgr_color(hsv):
    c = np.uint8([[list(hsv)]])
    c = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)
    return (int(c[0][0][0]), int(c[0][0][1]), int(c[0][0][2]))


if __name__ == '__main__':
    import os

    fileDir = os.path.dirname(os.path.realpath(__file__))
    vidDir = os.path.join(fileDir, '..', 'videos')
    imgDir = os.path.join(fileDir, '..', 'images')

    sampleImgPath = os.path.join(imgDir, 'test.jpg')

    img = cv2.imread(sampleImgPath)

    """
    img = get_crowdless_image(img)

    #dominantColorset = get_dominant_colorset(img, thresh=0.03, ignore_crowd = True, peak_num=3)
    #dominantColorset = get_dominant_colorset(img, thresh=0.02, ignore_crowd = False, peak_num=3)

    imgCpy = img.copy()

    # GSW jerseys
    lower = (115, 190, 80)
    upper = (125, 260, 260)



    # mask = get_jersey_mask(imgCpy, color1, color2)
    mask = get_jersey_mask(imgCpy, lower, upper)

    # grayMask = create_court_mask(imgCpy, dominantColorset, True)

    # cv2.imshow('mask', grayMask)
    """
    """
    dominantColorset1 = get_dominant_colorset(img, thresh=0.02, ignore_crowd = True, peak_num=1)
    dominantColorset2 = get_dominant_colorset(img, thresh=0.1, ignore_crowd = True, peak_num=2)
    dominantColorset3 = get_dominant_colorset(img, thresh=0.1, ignore_crowd = True, peak_num=4)
    mask1 = create_court_mask(img, dominantColorset1, True)
    mask2 = create_court_mask(img, dominantColorset2, True)
    mask3 = create_court_mask(img, dominantColorset3, True)

    cv2.imshow('mask1', mask1)
    cv2.imshow('mask2', mask2)
    cv2.imshow('mask3', mask3)
    """
    small = get_middle_court(img)
    cv2.imshow('small', small)
    jersey1 = get_jersey1_colors(img, thresh=0.8, peak_num=3)
    mask1 = create_court_mask(img, jersey1, True)
    cv2.imshow('mask1', mask1)
    cv2.imshow('original', img)



    cv2.waitKey(0)

