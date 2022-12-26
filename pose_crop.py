#!/usr/bin/env python
'''
The script contains a class for pose estimation on YOLOV7 model. 

The script takes an image and output from a YOLOV7 pose model and returns cropped
images of head, eyes, both feet, chest and hip.

Written by: Jubayer Hossain Ahad
'''

import time
import random
import os
import numpy as np
import cv2

__author__ = "Jubayer Hossain Ahad"
__maintainer__ = "Jubayer Hossain Ahad"
__email__ = "jubayerhossainahad@gmail.com"
__status__ = "Production"

class PoseCrop:
    def __init__(self):
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.eyes = [2, 3]
        self.hips = [12, 13]
        self.feet = [16, 17]
        self.chest = [6, 7]
        self.head = [1, 2, 3, 4, 5]
        self.elbows = [8, 9]
        self.knees = [14, 15]

    def coordinates(self, a, kpts, steps= 3,):
        conf = kpts[(a-1)*steps+2]
        if conf < 0.5:
            return None
        else:
            return (int(kpts[(a-1)*steps]), int(kpts[(a-1)*steps+1]))

    def get_chest_image(self, image, output):
        kpts = output[0, 7:].T
        p_8 = self.coordinates(8, kpts=kpts) # left elbow
        p_9 = self.coordinates(9, kpts=kpts) # right elbow

        p_6 = self.coordinates(6, kpts=kpts) # left shoulder
        p_7 = self.coordinates(7, kpts=kpts) # right shoulder

        top_most = image.shape[0]
        left_most = image.shape[1]
        bottom_most = 0
        right_most = 0

        if p_8 is None or p_9 is None or p_6 is None or p_7 is None:
            return None

        chest_elbow_distance = np.max([np.sqrt((p_8[0] - p_6[0])**2 + (p_8[1] - p_6[1])**2), np.sqrt((p_9[0] - p_7[0])**2 + (p_9[1] - p_7[1])**2)])

        # left_most and right_most
        if p_6[0] < p_7[0]:
            left_most = p_6[0] - (chest_elbow_distance/2)
            right_most = p_7[0] + (chest_elbow_distance/2)
        else:
            left_most = p_7[0] - (chest_elbow_distance/2)
            right_most = p_6[0] + (chest_elbow_distance/2)

        # top_most and bottom_most
        if p_6[1] < p_7[1]:
            top_most = p_6[1]
            bottom_most = p_7[1] + chest_elbow_distance
        else:
            top_most = p_7[1]
            bottom_most = p_6[1] + chest_elbow_distance

        if top_most < 0:
            top_most = 0
        if bottom_most > image.shape[0]:
            bottom_most = image.shape[0]
        if left_most < 0:
            left_most = 0
        if right_most > image.shape[1]:
            right_most = image.shape[1]

        chest_image = image[int(top_most):int(bottom_most), int(left_most):int(right_most)]
        return chest_image

    def get_hip_image(self, image, output):
        kpts = output[0, 7:].T
        p_12 = self.coordinates(12, kpts=kpts) # left hip
        p_13 = self.coordinates(13, kpts=kpts) # right hip

        p_14 = self.coordinates(14, kpts=kpts) # left knee
        p_15 = self.coordinates(15, kpts=kpts) # right knee

        top_most = image.shape[0]
        bottom_most = 0
        left_most = image.shape[1]
        right_most = 0

        if p_12 is None or p_13 is None or p_14 is None or p_15 is None:
            return None

        hip_knee_distance = np.max([np.sqrt((p_12[0] - p_14[0])**2 + (p_12[1] - p_14[1])**2), np.sqrt((p_13[0] - p_15[0])**2 + (p_13[1] - p_15[1])**2)])

        # left_most and right_most
        # left_most and right_most
        if p_12[0] < p_13[0]:
            left_most = p_12[0] - (hip_knee_distance/2)
            right_most = p_13[0] + (hip_knee_distance/2)
        else:
            left_most = p_13[0] - (hip_knee_distance/2)
            right_most = p_12[0] + (hip_knee_distance/2)

        # top_most and bottom_most
        if p_12[1] < p_13[1]:
            top_most = p_12[1] - (hip_knee_distance/2)
            bottom_most = p_13[1] + (hip_knee_distance/2)
        else:
            top_most = p_13[1] - (hip_knee_distance/2)
            bottom_most = p_12[1] + (hip_knee_distance/2)

        if top_most < 0:
            top_most = 0
        if bottom_most > image.shape[0]:
            bottom_most = image.shape[0]
        if left_most < 0:
            left_most = 0
        if right_most > image.shape[1]:
            right_most = image.shape[1]

        hip_image = image[int(top_most):int(bottom_most), int(left_most):int(right_most)]
        return hip_image

    def get_head_image(self, image, output):
        image = image
        output = output
        kpts = output[0, 7:].T
        p_1 = self.coordinates(1, kpts=kpts) # nose
        p_6 = self.coordinates(6, kpts=kpts) # left shoulder
        p_7 = self.coordinates(7, kpts=kpts) # right shoulder

        if p_1 is None or p_6 is None or p_7 is None:
            return None

        max_nose2shoulder = np.max([np.sqrt((p_1[0] - p_6[0])**2 + (p_1[1] - p_6[1])**2), np.sqrt((p_1[0] - p_7[0])**2 + (p_1[1] - p_7[1])**2)])

        top_most = image.shape[0]
        bottom_most = 0
        left_most = image.shape[1]
        right_most = 0

        left_most = p_1[0] - max_nose2shoulder
        right_most = p_1[0] + max_nose2shoulder
        top_most = p_1[1] - max_nose2shoulder
        bottom_most = p_1[1] + max_nose2shoulder

        if top_most < 0:
            top_most = 0
        if bottom_most > image.shape[0]:
            bottom_most = image.shape[0]
        if left_most < 0:
            left_most = 0
        if right_most > image.shape[1]:
            right_most = image.shape[1]

        head_image = image[int(top_most):int(bottom_most), int(left_most):int(right_most)]
        return head_image

    def get_eye_image(self, image, output):
        image = image
        output = output
        kpts = output[0, 7:].T
        p_2 = self.coordinates(2, kpts=kpts) # left eye
        p_3 = self.coordinates(3, kpts=kpts) # right eye

        if p_2 is None or p_3 is None:
            return None

        eye_distance = np.sqrt((p_2[0] - p_3[0])**2 + (p_2[1] - p_3[1])**2)

        top_most = image.shape[0]
        bottom_most = 0
        left_most = image.shape[1]
        right_most = 0

        # left_most and right_most
        if p_2[0] < p_3[0]:
            left_most = p_2[0] - eye_distance
            right_most = p_3[0] + eye_distance
        else:
            left_most = p_3[0] - eye_distance
            right_most = p_2[0] + eye_distance

        # top_most and bottom_most
        if p_2[1] < p_3[1]:
            top_most = p_2[1] - eye_distance
            bottom_most = p_3[1] + eye_distance
        else:
            top_most = p_3[1] - eye_distance
            bottom_most = p_2[1] + eye_distance

        if top_most < 0:
            top_most = 0
        if bottom_most > image.shape[0]:
            bottom_most = image.shape[0]
        if left_most < 0:
            left_most = 0
        if right_most > image.shape[1]:
            right_most = image.shape[1]

        eye_image = image[int(top_most):int(bottom_most), int(left_most):int(right_most)]
        return eye_image

    def get_feet_image(self, image, output):
        kpts = output[0, 7:].T
        # foot crop
        p_14 = self.coordinates(14, kpts=kpts) # left knee
        p_15 = self.coordinates(15, kpts=kpts) # right knee

        p_16 = self.coordinates(16, kpts=kpts) # left ankle
        p_17 = self.coordinates(17, kpts=kpts) # right ankle

        if p_14 is None or p_15 is None or p_16 is None or p_17 is None:
            return None, None

        left_foot_dist = np.sqrt((p_14[0] - p_16[0])**2 + (p_14[1] - p_16[1])**2)
        right_foot_dist = np.sqrt((p_15[0] - p_17[0])**2 + (p_15[1] - p_17[1])**2)

        # left foot crop
        top_most = image.shape[0]
        bottom_most = 0
        left_most = image.shape[1]
        right_most = 0

        left_most = p_16[0] - (left_foot_dist/2)
        right_most = p_16[0] + (left_foot_dist/2)
        top_most = p_16[1] - (left_foot_dist/2)
        bottom_most = p_16[1] + (left_foot_dist/2)

        if top_most < 0:
            top_most = 0
        if bottom_most > image.shape[0]:
            bottom_most = image.shape[0]
        if left_most < 0:
            left_most = 0
        if right_most > image.shape[1]:
            right_most = image.shape[1]

        # crop image
        left_foot_img = image[int(top_most):int(bottom_most), int(left_most):int(right_most)]

        # right foot crop
        top_most = image.shape[0]
        bottom_most = 0
        left_most = image.shape[1]
        right_most = 0

        left_most = p_17[0] - (right_foot_dist/2)
        right_most = p_17[0] + (right_foot_dist/2)
        top_most = p_17[1] - (right_foot_dist/2)
        bottom_most = p_17[1] + (right_foot_dist/2)

        if top_most < 0:
            top_most = 0
        if bottom_most > image.shape[0]:
            bottom_most = image.shape[0]
        if left_most < 0:
            left_most = 0
        if right_most > image.shape[1]:
            right_most = image.shape[1]

        # crop image
        right_foot_img = image[int(top_most):int(bottom_most), int(left_most):int(right_most)]

        return left_foot_img, right_foot_img

    