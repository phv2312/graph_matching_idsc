import numpy as np
import cv2
from skimage import measure
from scipy.spatial.distance import cdist

from stuffs.IDSC.shape_context import ShapeContext
from stuffs.gaussianfield import gaussianfield

class ComponentUtils:
    def __init__(self, dist_per_point=10):
        # Segmentation
        self.bad_values = [x + 300 * (x + 1) + 300 * 300 * (x + 1) for x in [0, 5, 10, 15, 255]]
        self.min_area = 50
        self.min_size = 3

        # Descriptor
        self.shape_descriptor = ShapeContext(nbins_r=5, nbins_theta=12, r_inner=0.125, r_outer=2.)
        self.dist_per_point = dist_per_point

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
                components[index]['idsc'] = self.calc_feat(components[index]['image'])

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
            components[index]['idsc'] = self.calc_feat(components[index]['image'])

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
        points = self.shape_descriptor.get_points_from_img(component_image, self.dist_per_point)
        feats  = self.shape_descriptor.compute(points)

        return (points, feats)

    def compare_feats(self, source_components, target_components, threshold=0.2):
        K = np.zeros(shape=(len(source_components), len(target_components)), dtype=np.float32)

        for s_i, s_component in enumerate(source_components):
            for t_i, t_component in enumerate(target_components):
                K[s_i, t_i] = self.compare_feat(s_component['idsc'], t_component['idsc'], threshold)

        return K

    def compare_feat(self, source_idsc, target_idsc, threshold=0.2):

        source_points, source_feats = source_idsc
        target_points, target_feats = target_idsc

        source_feat = source_feats.flatten().reshape(1,-1)
        target_feat = target_feats.flatten().reshape(1,-1)

        # find num matching
        point_dists = cdist(source_feats, target_feats, metric='cosine')
        n_matching = np.sum(point_dists <= threshold)

        # calculate distance
        feat_dist = cdist(source_feat, target_feat, metric='cosine')
        print (feat_dist, n_matching, len(source_points), len(target_points))
        return feat_dist / (n_matching + 1e-6)

"""
field_solution, inverse_laplacian = gaussianfield.solve(K, labels[observed], observed)
class_predictions = np.argmax(field_solution, axis=1)

# compute expected risk for active learning...
risk = gaussianfield.expected_risk(field_solution, inverse_laplacian)

# get the query index relative to the full dataset
_risk = 1000 * np.ones(labels.shape[0])
_risk[~observed] = risk
query_idx = np.argmin(_risk)

print ('query_idx:', query_idx)
"""

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
    pass
    # plt.imshow(im)
    # plt.show()






