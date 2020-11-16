import numpy as np
import cv2
from skimage import measure
from stuffs.gaussianfield import gaussianfield
from IDSC.IDSC import IDSCDescriptor, matching, calc_matching_distance


def calc_softmax(X, dim):
    x_e = np.exp(X)

    out = x_e / np.sum(x_e, axis=dim, keepdims=True)
    return out

class ComponentUtils:
    def __init__(self, max_contour_points=100):
        # Segmentation
        self.bad_values = [x + 300 * (x + 1) + 300 * 300 * (x + 1) for x in [0, 5, 10, 15, 255]]
        self.min_area = 50
        self.min_size = 3

        # Descriptor
        self.shape_descriptor = IDSCDescriptor(max_contour_points=max_contour_points)

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
                    "bbox": region.bbox
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
                "bbox": region.bbox
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

    def compare_feats(self, source_components, target_components, penalty=0.3, alpha=10.):
        # consesus matching ...
        K = np.zeros(shape=(len(source_components), len(target_components)), dtype=np.float32)
        K_inv = np.zeros(shape=(len(target_components), len(source_components)), dtype=np.float32)

        for s_i, s_component in enumerate(source_components):
            for t_i, t_component in enumerate(target_components):
                k_v, k_v_inv = self._compare_feat(s_component['idsc'], t_component['idsc'], penalty)
                K[s_i, t_i] = k_v
                K_inv[t_i, s_i] = k_v_inv

        # convert distance to score
        K = 1. / (1. + K)
        K_inv = 1. / (1. + K_inv)

        K = K - np.max(K, axis=1, keepdims=True)
        K_inv = K_inv - np.max(K_inv, axis=1, keepdims=True)

        K_sm = calc_softmax(alpha * K, dim=1)
        K_inv_sm = calc_softmax(K_inv, dim=1)

        K = np.sqrt(K_sm * K_inv_sm.T)

        return K

    def _compare_feat(self, source_idsc, target_idsc, penalty=0.3):

        source_feats, source_points = source_idsc
        target_feats, target_points = target_idsc

        # get pair matching and cost
        pair_ids, _, _, _, _, dist_matrix = matching(source_feats,
                                                    target_feats, 
                                                    source_points, 
                                                    target_points)

        matching_cost_s2t = calc_matching_distance(dist_matrix, pair_ids, penalty=penalty)
        matching_cost_t2s = calc_matching_distance(dist_matrix.T, pair_ids=[(t,s) for (s,t) in pair_ids], penalty=penalty)

        return matching_cost_s2t, matching_cost_t2s

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






