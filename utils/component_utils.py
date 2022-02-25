import cv2
import numpy as np
from skimage import measure


def extract_skimage_components(image, background_value=0):
    """
    Extracting connected components
    :param image: binary image
    :param background_value: value of pixel to be considered as background
    """
    h, w = image.shape[0], image.shape[1]
    result_dct = []

    label = measure.label(image, neighbors=4, background=background_value)
    for region in measure.regionprops(label):
        area = region['area']
        centroid = region['centroid']
        ys, xs = region['coords'][:, 0], region['coords'][:, 1]

        image_org_size = np.zeros(shape=(h, w), dtype=np.uint8)
        image_org_size[ys, xs] = 1

        result_dct += [{
            'area': area,
            'centroid': centroid,
            'image_org_size': image_org_size,
            'coord': [xs, ys]
        }]

    return result_dct


def transfer_color(reference_image_color, target_image_gray, character_reference, character_target, matching):
    """
    Transfer the color from reference to target
    :param reference_image_color: rgb color image
    :param target_image_gray: gray image
    :param character_reference: instance of class Character
    :param character_target: instance of class Character
    :param matching: list of tuple(reference_id, target_id)
    """

    target_image_color = cv2.cvtColor(target_image_gray, cv2.COLOR_GRAY2RGB)
    for reference_id, target_id in matching:
        node_reference = character_reference.nodes[reference_id]
        node_target = character_target.nodes[target_id]

        xs_reference, ys_reference = node_reference.coord
        xs_target, ys_target = node_target.coord
        target_image_color[ys_target, xs_target] = reference_image_color[ys_reference[0], xs_reference[0], :]
    return target_image_color

