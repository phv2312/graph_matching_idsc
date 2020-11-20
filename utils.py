import numpy as np
import cv2
from skimage import measure
from libs.gaussianfield import gaussianfield
from libs.feature_extract.IDSC import IDSCDescriptor, matching, calc_matching_distance
from scipy.spatial.distance import cdist
from functools import partial
import multiprocessing
import time

MULTI_PROCESS = True

def calc_softmax(X, dim, alpha):
    X_ = X - np.max(X, dim, keepdims=True)
    x_e = np.exp(alpha * X_)
    out = x_e / np.sum(x_e, axis=dim, keepdims=True)
    return out

def _calc_distance_single(f1, f2, min_matching_threshold, penalty):
    feats1, points1 = f1[:, :-2], f1[:, -2:]
    feats2, points2 = f2[:, :-2], f2[:, -2:]

    pair_ids, _, _, _, _, dist_matrix = matching(feats1,
                                                 feats2,
                                                 points1,
                                                 points2,
                                                 min_threshold=min_matching_threshold)

    matching_cost_s2t = calc_matching_distance(dist_matrix, pair_ids, penalty=penalty)

    return matching_cost_s2t, len(pair_ids)


class ComponentUtils:
    def __init__(self, max_contour_points=100, n_angle_bins=8, n_distance_bins=8):
        # Segmentation
        self.bad_values = [x + 300 * (x + 1) + 300 * 300 * (x + 1) for x in [0, 5, 10, 15, 255]]
        self.min_area = 50
        self.min_size = 3

        # Descriptor
        self.shape_descriptor = IDSCDescriptor(max_contour_points, n_angle_bins, n_distance_bins)

    def __extract_on_color_image(self, input_image):
        b, g, r = cv2.split(input_image)
        b, g, r = b.astype(np.uint64), g.astype(np.uint64), r.astype(np.uint64)

        index = 0
        components = {}
        mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.int)

        # Pre-processing image
        processed_image = b + 300 * (g + 1) + 300 * 300 * (r + 1)
        # Get number of colors in image
        uniques = np.unique(processed_image)

        for unique in uniques:
            # Ignore sketch (ID of background is 255)
            if unique in self.bad_values:
                continue

            rows, cols = np.where(processed_image == unique)
            # Mask
            image_temp = np.zeros_like(processed_image)
            image_temp[rows, cols] = 255
            image_temp = np.array(image_temp, dtype=np.uint8)

            # Connected components
            labels = measure.label(image_temp, connectivity=1, background=0)
            regions = measure.regionprops(labels, intensity_image=processed_image)

            for region in regions:
                if region.area < self.min_area:
                    continue
                if abs(region.bbox[2] - region.bbox[0]) < self.min_size:
                    continue
                if abs(region.bbox[3] - region.bbox[1]) < self.min_size:
                    continue

                if unique == 23117055 and [0, 0] in region.coords:
                    continue

                components[index] = {
                    "centroid": np.array(region.centroid),
                    "area": region.area,
                    "image": cv2.copyMakeBorder(region.image.astype(np.uint8) * 255, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0),
                    "label": index + 1,
                    "coords": region.coords,
                    "bbox": region.bbox,
                    "mask": index + 1
                }
                components[index]['idsc'] = self.calc_feat(components[index]['image'].copy())

                mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
                index += 1

        components = [components[i] for i in range(0, len(components))]
        return mask, components

    def __extract_on_sketch(self, input_image):
        if len(input_image.shape) > 2:
            binary = cv2.threshold(input_image, 100, 255, cv2.THRESH_BINARY)[1]
        else:
            binary = input_image

        labels = measure.label(binary, connectivity=1, background=0)
        regions = measure.regionprops(labels, intensity_image=input_image)

        index = 0
        mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.int)
        components = dict()

        for region in regions[1:]:
            if region.area < self.min_area:
                continue
            if abs(region.bbox[2] - region.bbox[0]) < self.min_size:
                continue
            if abs(region.bbox[3] - region.bbox[1]) < self.min_size:
                continue

            components[index] = {
                "centroid": np.array(region.centroid),
                "area": region.area,
                "image": cv2.copyMakeBorder(region.image.astype(np.uint8) * 255, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0),
                "label": index + 1,
                "coords": region.coords,
                "bbox": region.bbox,
                "mask": index + 1
            }
            components[index]['idsc'] = self.calc_feat(components[index]['image'].copy())

            mask[region.coords[:, 0], region.coords[:, 1]] = index + 1
            index += 1

        components = [components[i] for i in range(0, len(components))]
        return mask, components

    def extract_component(self, input_image, mode):
        assert mode in ('color', 'sketch')

        if mode == 'color':
            mask, components = self.__extract_on_color_image(input_image)
        else:
            mask, components = self.__extract_on_sketch(input_image)

        return mask, components

    def calc_feat(self, component_image):
        # component_image is in 0-background, 1-foreground
        # must return the (sampling point locations, feature for each points)
        # component_image is binary image
        points, feats = self.shape_descriptor.describe(component_image)
        return (points, feats)

    def compare_feats_consensus(self, source_components, target_components, penalty=0.3, min_threshold_area=0.35):
        """
        return matrix k: of size (src, tgt), which means, for each source, find target
        """

        #
        K1, (K2, K2_min, K2_max) = \
            self.compare_feats(source_components, target_components, penalty, min_threshold_area) # (s,t)
        K1_,(K2_, K2_min_, K2_max_) = \
            self.compare_feats(target_components, source_components, penalty, min_threshold_area) # (t,c)

        #
        K1 = 1 / (1 + K1)
        K1_ = 1 / (1 + K1_)

        alpha1, alpha1_ = 1. ,1.
        K1_ = calc_softmax(K1_.T, dim=1, alpha=alpha1)
        K1  = calc_softmax(K1, dim=1, alpha=alpha1_)
        K1  = K1 * K1_

        #
        K2  = np.sqrt(K2 * K2_.T)

        #
        K2_min = np.sqrt(K2_min * K2_min_.T)

        #
        K2_max = np.sqrt(K2_max * K2_max_.T)

        return K1, (K2, K2_min, K2_max)

    def compare_feats(self, source_components, target_components, penalty=0.3, min_threshold_area=0.35):
        def _get_feature(c):
            feats, points = c['idsc']
            return np.concatenate([feats, points], axis=1)

        source_feats = np.array([_get_feature(c) for c in source_components], dtype=np.object)
        target_feats = np.array([_get_feature(c) for c in target_components], dtype=np.object)
        custom_distance = partial(_calc_distance_single, min_matching_threshold=min_threshold_area, penalty=penalty)

        all_ids = [(si,ti) for si in range(source_feats.shape[0]) for ti in range(target_feats.shape[0])]

        if MULTI_PROCESS:
            with multiprocessing.Pool(processes=8) as p:
                result = p.starmap(custom_distance, [(source_feats[si], target_feats[ti]) for (si, ti) in all_ids])
        else:
            result = [custom_distance(source_feats[si], target_feats[ti]) for (si, ti) in all_ids]

        result = np.array(result)
        K = np.zeros(shape=(source_feats.shape[0], target_feats.shape[0], 2), dtype=np.float32)

        s_ids = [ids[0] for ids in all_ids]
        t_ids = [ids[1] for ids in all_ids]
        K[s_ids, t_ids] = result

        K1 = K[:,:,0].astype(np.float32) # distance
        K2 = K[:,:,1].astype(np.int32) # n_matching point

        K3 = K2 / np.array([len(f) for f in source_feats]).reshape(-1, 1)
        K4 = K2 / np.array([len(f) for f in target_feats]).reshape(1, -1)

        K21 = np.stack([K3, K4], axis=-1)
        K2_max = np.max(K21, axis=-1)
        K2_min = np.min(K21, axis=-1)

        return K1, (K2, K2_min, K2_max)

class ALUtils:
    def estimate_risk(self, K, labels, observed):
        field_solution, inverse_laplacian = gaussianfield.solve(K, labels[observed], observed)
        risk = gaussianfield.expected_risk(field_solution, inverse_laplacian)

        _risk = gaussianfield.expected_risk(field_solution, inverse_laplacian)
        _risk[~observed] = risk

        return _risk

    def estimate_label(self, K, labels, observed):
        field_solution, inverse_laplacian = gaussianfield.solve(K, labels[observed], observed)
        return field_solution

import matplotlib.pyplot as plt
def imgshow(im):
    #pass
    plt.imshow(im)
    plt.show()






