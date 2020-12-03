# encoding: utf-8
import os
import cv2
import copy
import math
import numpy as np
import pandas as pd
from skimage import morphology
from skimage import draw
from skimage import measure
from skimage.measure import find_contours
from skimage.morphology import erosion
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from itertools import combinations
from enum import Enum, unique

@unique
class LesionQuality(Enum):
    Solid = 0
    Cystic = 1
    Spongiform = 2
    Mixed_Solid = 3
    Mixed_Cystic = 4


class ThyroidFeature(object):
    def __init__(self, image_id, image, mask):
        self.id = image_id
        self.image = image
        self.mask = (mask/255).astype(np.uint8)
        self.contour = self.get_contour()
        self.roi, self.roi_mask = self.get_lesion_roi()
        
    def get_contour(self):
        contour_obj, _ = cv2.findContours(self.mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contour_obj[0]

    def get_lesion_roi(self, th=8):
        x, y, w, h = cv2.boundingRect(self.contour)
        y0, y1, x0, x1 = x - th, x + w + th, y - th, y + h + th
        x0 = max(x0,0)
        y0 = max(y0,0)
        x1 = min(x1,self.image.shape[0]-1)
        y1 = min(y1,self.image.shape[1]-1)
        img = self.image[x0:x1, y0:y1]
        mask = (self.mask[x0:x1, y0:y1]).astype(np.uint8)
        return img, mask

    def compute_aspect_ratio(self, th_angle=(45, 145)):
        """
            Compute aspect_ratio
            :param th_angle: angle_range
            :return: aspect_ratio
        """
        major_axis, minor_axis = self.get_lesion_axis()
        l_points = major_axis[1]
        px0, py0, px1, py1 = l_points[0][0], l_points[0][1], l_points[1][0], l_points[1][1]
        angle = math.atan2((px1 - px0), (py1 - py0))
        theta = abs(angle * (180 / np.pi))
        if theta > th_angle[0] and theta < th_angle[1]:
            return minor_axis[0]/major_axis[0]
        else:
            return major_axis[0]/minor_axis[0]

    def analyze_quality(self):
        """
        :return: one of [Solid, Cystic, Spongiform, Mixed_Solid, Mixed_Cystic]
        """
        cystic_ratio, count_cystic_region = self.get_percentage_no_echo(25)

        solid_ratio = 1 - cystic_ratio
        if cystic_ratio > 0.95:
            return LesionQuality.Cystic
        elif solid_ratio > 0.95:
            return LesionQuality.Solid
        elif solid_ratio > 0.5:
            return LesionQuality.Mixed_Solid
        elif count_cystic_region > 3:
            return LesionQuality.Spongiform
        else:
            return LesionQuality.Mixed_Cystic

    def compute_echo(self):
        th = self.compute_threshold()
        echo_img = copy.deepcopy(self.roi)

        echo_img[self.roi >= th] = 255
        echo_img[self.roi < th] = 0

        roi = self.roi_mask * echo_img
        ver_roi = self.roi_mask * (1 - echo_img/255) * 255
        prct_no_echo, _ = self.get_percentage_no_echo()
        prct_equal_echo = self.get_percentage_equal_echo(roi)
        prct_low_echo = self.get_percentage_equal_echo(ver_roi) - prct_no_echo
        composition = {
            'prct_no_echo': prct_no_echo,
            'prct_low_echo': prct_low_echo,
            'prct_equal_echo': prct_equal_echo
        }
        
        echo_img[echo_img == 0] = 120
        echo_img[self.roi < 20] = 0
        echo_img = echo_img*self.roi_mask

        return echo_img, composition

    def get_percentage_equal_echo(self, roi):
        mask_sum = float(np.sum(self.roi_mask))
        roi_sum = np.sum(roi/255)
        echo_ratio = roi_sum/mask_sum
        return round(echo_ratio, 2)
    
    def get_percentage_no_echo(self, th=20):
        roi_img = copy.deepcopy(self.roi)
        roi_img[self.roi < th] = 255
        roi_img[roi_img != 255] = 0
        roi_img = roi_img * self.roi_mask
        prct_no_echo = self.get_percentage_equal_echo(roi_img)
        
        # count no_echo_region
        count_no_echo_region = ThyroidFeature.analyze_num(roi_img)
        return prct_no_echo, count_no_echo_region

    def compute_threshold(self):
        roi = cv2.GaussianBlur(self.roi, (3, 3), 0)
        roi_img, threshold = self.process_background(roi)
        roi_img[roi_img > 150] = 0
        roi_img[roi_img < 20] = 0
        ret_otsu, _ = cv2.threshold(roi_img[roi_img != 0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return (threshold + ret_otsu)/2

    @staticmethod
    def analyze_num(roi):
        mask, count = measure.label(roi, connectivity=1, return_num=True)
        num = 0
        roi_sum = float(np.sum(roi/255))
        for region in measure.regionprops(mask):
            area_ratio = region.area / roi_sum
            if area_ratio > 0.05:
                num += 1
        return num

    def process_background(self, roi):
        img = copy.deepcopy(roi)
        # get background
        roi = img * (1 - self.roi_mask)
        # get mode
        counts = np.bincount(roi[roi > 20])
        g_med = np.argmax(counts)
        # get average
        roi[roi > (g_med + 30)] = 0
        roi[roi < (g_med - 30)] = 0
        g_med = np.mean(roi[roi != 0])
        img = (1 - self.roi_mask) * roi + self.roi_mask * self.roi
        # cap g_med
        if g_med > 95:
            g_med = 95
        elif g_med < 45:
            g_med = 45

        return img, g_med - 5

    def get_lesion_axis(self):
        """
        compute major axis and minor axis
        :return:
             major axis
             minor axis
        """

        def __dot_product(pair1, pair2):
            """
            compute dot product
            :param pair1: tuple: [[x1,y1],[x2,y2]]
            :param pair2: tuple: [[x3,y3],[x4,y4]]
            :return:
              dot: non-negative int
            """
            v1 = [i - j for i, j in zip(pair1[0], pair1[1])]
            v2 = [i - j for i, j in zip(pair2[0], pair2[1])]
            dot = abs(sum(x * y for x, y in zip(v1, v2)))
            return dot

        def __get_major_axis(contour):
            """
            Compute major axis: [axis_length, tuple(coordinate1, coordinate2)]
            :param contour: list [[x1,y1],[x2,y2]....]
            :return:
                result = [float, tuple(coordinate, coordinate)]
            """
            axis_coordinate_pair = 0
            max_length = 0
            # Find max distance between all at coordinate pairs on contour
            for pair in combinations(contour, 2):
                length = np.linalg.norm(pair[0]-pair[1])
                if length > max_length:
                    axis_coordinate_pair = pair
                    max_length = length

            # major_axis
            axis_length = np.linalg.norm(axis_coordinate_pair[0]-axis_coordinate_pair[1])

            major_axis = [axis_length, axis_coordinate_pair]

            return major_axis

        def __get_minor_axis(major_axis_coordinate_pair, contour):
            """
            Compute minor axis: [axis_length, tuple(coordinate1, coordinate2)]
            :param contour: list [[x1,y1],[x2,y2]....]
            :return:
                result = [float, tuple(coordinate, coordinate)]
            """

            pair_list = []
            minor_axis_length_list = []
            dot_product_threshold = 15

            # get possible minor axis
            for pair in combinations(contour, 2):
                dot_product = __dot_product(pair, major_axis_coordinate_pair)
                if dot_product >= dot_product_threshold:
                    continue
                pair_list.append(pair)
                length = np.linalg.norm(pair[0]-pair[1])
                minor_axis_length_list.append(length)

            # no possible minor axis, use mid point of amjor axis
            if len(minor_axis_length_list) == 0:  
                major_axis_indices = [i for i in range(len(contour)) if contour[i] in major_axis_coordinate_pair]
                major_axis_idx_1 = min(major_axis_indices)
                major_axis_idx_2 = max(major_axis_indices)
                minor_axis_idx_1 = (major_axis_idx_1 + major_axis_idx_2) / 2

                minor_axis_idx_2 = (len(contour) - 1 - major_axis_idx_2 + major_axis_idx_1) / 2 + major_axis_idx_2
                if minor_axis_idx_2 >= len(contour):
                    minor_axis_idx_2 = major_axis_idx_1 - (len(contour) - 1 - major_axis_idx_2 + major_axis_idx_1) / 2

                minor_axis_coordinate_pair = contour[sd_idx_1], contour[sd_idx_2]
                minor_axis_length = np.linalg.norm(minor_axis_coordinate_pair[0]-minor_axis_coordinate_pair[1])
            else:
                minor_axis_length = max(minor_axis_length_list)
                minor_axis_index = [i for i, j in enumerate(minor_axis_length_list) if j == minor_axis_length]
                # if multiple possible minor axis, chooose one
                minor_axis_coordinate_pair = pair_list[minor_axis_index[0]]

            result = [minor_axis_length, minor_axis_coordinate_pair]
            return result

        major_axis = __get_major_axis(self.contour[:, 0])
        minor_axis = __get_minor_axis(major_axis[1], self.contour[:, 0])

        return major_axis, minor_axis


