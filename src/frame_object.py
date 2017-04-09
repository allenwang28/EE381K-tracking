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

class FrameObject():
    # Variables
    _frame_num = None
    _video_title = None
    # Images
    _bgr_img = None
    _gray_mask = None
    _dominant_colorset = None
    _gray_flooded2 = None
    _away_colors = None
    _away_mask = None
    # Lines
    _sideline = None
    _baseline = None
    _freethrowline = None
    _close_paintline = None
    # Booleans
    _sideline_detected = True
    _baseline_detected = True
    _freethrowline_detected = True
    _close_paintline_detected = True

    # Points
    _sideline_baseline = None # The far one in the corner
    _close_paint_baseline = None # Intersection between close paintline and baseline
    _close_paint_freethrow = None # Int btw close paintline and freethrow line
    _sideline_freethrow = None # Int btw far sideline and freethrow line

    _away_player_coordinates = None

    def __init__(self, img, frame_num, video_title, away_colors):
        assert img is not None
        self._bgr_img = img
        self._frame_num = frame_num
        self._video_title = video_title
        self._away_colors = away_colors

    # Exported methods
    def get_gray_mask(self):
        if self._gray_mask is None:
            d_c = self.get_dominant_colorset()
            self._gray_mask = \
                colors.create_court_mask(self.get_bgr_img(), d_c, True)
        return self._gray_mask.copy()


    def get_bgr_img(self):
        return self._bgr_img.copy()


    def get_dominant_colorset(self):
        if self._dominant_colorset is None:
            self._dominant_colorset = colors.get_dominant_colorset(self.get_bgr_img())
        # colors.show_hist(self._dominant_colorset)
        return self._dominant_colorset.copy()


    def get_gray_flooded2(self):
        if self._gray_flooded2 is None:
            self._gray_flooded2 = \
                colors.get_double_flooded_mask(self.get_gray_mask())
        return self._gray_flooded2.copy()


    def get_sideline(self):
        if self._sideline is None:
            lines = tld.find_top_boundary(self.get_gray_mask())
            print("Lines----")
            print type(lines)
            print lines
            # img = colors.gray_to_bgr(self._gray_mask.copy())
            # hough.put_lines_on_img(img, lines)
            # cv2.imwrite('images/6584_toplines.jpg', img)
            if len(lines) < 2:
                _baseline_detected = False
                print "Baseline not detected"
            self._sideline = lines[0]
            self._baseline = lines[1]
        return self._sideline


    def get_baseline(self):
        if self._baseline is None:
            _ = self.get_sideline()
        return self._baseline


    def get_freethrowline(self):
        if self._freethrowline is None:
            lines = hough.get_lines_from_paint(self.get_gray_flooded2(),
                self.get_sideline(), self.get_baseline(), verbose=True)
            if lines[0] is None:
                self._freethrowline_detected = False
            if lines[1] is None:
                self._close_paintline_detected = False
            self._freethrowline, self._close_paintline = lines
        return self._freethrowline


    def get_close_paintline(self):
        if self._close_paintline is None:
            _ = self.get_freethrowline()
        return self._close_paintline


    def get_quadrangle_points(self):
        pts = []
        pts.append(get_intersection(self.get_sideline(), self.get_freethrowline()))
        pts.append(get_intersection(self.get_sideline(), self.get_baseline()))
        pts.append(get_intersection(self.get_close_paintline(), self.get_baseline()))
        pts.append(get_intersection(self.get_close_paintline(), self.get_freethrowline()))
        return pts


    def get_away_jersey_mask(self):
        if self._away_mask is None:
            crowdlessImg = colors.get_crowdless_image(self.get_bgr_img())
            self._away_mask = colors.get_away_jersey_mask(crowdlessImg, self._away_colors[0], self._away_colors[1])
        return self._away_mask

    def get_away_player_coordinates(self):
        if self._away_player_coordinates is None:
            self._away_player_coordinates = cluster.getCentroids(self.get_away_jersey_mask())
            self._away_player_coordinates = colors.remap_from_crowdless_coords(self.get_bgr_img(), self._away_player_coordinates)
        return self._away_player_coordinates

    def show_away_player_coordinates(self):
        coordinates = self.get_away_player_coordinates()
        circleImg = self.get_bgr_img()
        print coordinates
        for (x,y) in coordinates:
            circs = cv2.circle(circleImg, (x,y), 5, (255, 255, 255), -1)
            print x
            print y
        cv2.imshow('away players', circleImg)



    def show_lines(self):
        lines = [self.get_freethrowline(), self.get_close_paintline(),
            self.get_sideline(), self.get_baseline()]
        if not self.all_lines_detected():
            raise Exception("Not all lines were detected. Undetected: {}".format(self.get_undetected_lines()))
        img = colors.gray_to_bgr(self.get_gray_flooded2())
        hough.put_lines_on_img(img, lines)
        cv2.imshow('lines', img)

    def show_points(self):
        lines = [self.get_freethrowline(), self.get_close_paintline(),
            self.get_sideline(), self.get_baseline()]
        if not self.all_lines_detected():
            raise Exception("Not all lines were detected. Undetected: {}".format(self.get_undetected_lines()))
        img = self.get_bgr_img()
        hough.put_lines_on_img(img, lines)
        points = self.get_quadrangle_points()
        print (points)
        hough.put_points_on_img(img, points)
        cv2.imshow('points', img)


    def freethrowline_detected(self):
        return self._freethrowline_detected

    def baseline_detected(self):
        return self._baseline_detected

    def sideline_detected(self):
        return self._sideline_detected

    def close_paintline_detected(self):
        return self._close_paintline_detected

    def all_lines_detected(self):
        return self._freethrowline_detected and \
               self._baseline_detected and \
               self._close_paintline_detected and \
               self._sideline_detected

    def get_frame_number(self):
        return self._frame_num

    def get_video_title(self):
        return self._video_title

    def get_undetected_lines(self):
        undetected_lines = ""
        if not self._freethrowline_detected:
            undetected_lines += "free throw\n"
        if not self._baseline_detected:
            undetected_lines += "baseline\n"
        if not self._close_paintline_detected:
            undetected_lines += "paintline\n"
        if not self._sideline_detected:
            undetected_lines += "sideline\n"
        return undetected_lines


def testlines(img_obj, save_filename):
    lines = [img_obj.get_freethrowline(), img_obj.get_close_paintline(),
        img_obj.get_sideline(), img_obj.get_baseline()]
    img = colors.gray_to_bgr(img_obj.get_gray_flooded2())
    hough.put_lines_on_img(img, lines)
    cv2.imshow('lines', img)


def testpoints(img_obj, save_filename):
    lines = [img_obj.get_freethrowline(), img_obj.get_close_paintline(),
        img_obj.get_sideline(), img_obj.get_baseline()]
    img = img_obj.get_bgr_img()
    hough.put_lines_on_img(img, lines)
    points = img_obj.get_quadrangle_points()
    print (points)
    hough.put_points_on_img(img, points)
    cv2.imwrite(save_filename, img)


if __name__ == '__main__':
    import os
    fileDir = os.path.dirname(os.path.realpath(__file__))
    imgDir = os.path.join(fileDir, '..', 'images')

    sampleImgPath = os.path.join(imgDir, 'test.jpg')
    GSW_JERSEY_UPPER = (115, 190, 80)
    GSW_JERSEY_LOWER = (125, 260, 260)

    img = cv2.imread(sampleImgPath)
    
    fo = FrameObject(img, 0, '', (GSW_JERSEY_UPPER, GSW_JERSEY_LOWER))

    fo.show_away_player_coordinates()

    cv2.waitKey(0)


    


