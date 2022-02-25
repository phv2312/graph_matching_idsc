import os
import cv2
import numpy as np
from utils.component_utils import extract_skimage_components
from libs.feature_extract.IDSC import IDSCDescriptor


class Node:
    def __init__(self, area, centroid, image_org_size, coord):
        self.area = area  # float

        self.centroid = np.array(centroid)  # (2,)
        self.image_org_size = image_org_size
        self.feature = None
        self.coord = coord

    def calculate_idsc(self):
        descriptor = IDSCDescriptor(max_contour_points=50, n_angle_bins=12, n_distance_bins=12)
        self.feature = descriptor.describe(self.image_org_size)[0]

    def get_image(self):
        return self.image_org_size


class Character:
    def __init__(self, image, n_padding=0, min_area=100):
        self.image = image
        self.nodes = []  # list of Node
        self.n_padding = n_padding

        # nodes
        skimage_component_list = extract_skimage_components(image, background_value=0)
        for c in skimage_component_list:
            is_accepted = True

            if c['area'] < min_area:
                is_accepted = False

            if is_accepted:
                node = Node(**c)
                node.calculate_idsc()
                self.nodes += [node]

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, item):
        return self.nodes[item]
