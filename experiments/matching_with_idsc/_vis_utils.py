import numpy as np
import os
import cv2
from itertools import combinations
from PIL import Image

def build_binary_combination(input_list):
    res = []
    for x, y in combinations(input_list, 2):
        res.append([x, y])

    return res

def get_mask(region, input_image):
    mask = np.zeros((input_image.shape[0], input_image.shape[1]), dtype=np.int)
    mask[region[:, 0], region[:, 1]] = 255

    return mask

def component_extract(folder_matching, src_img_name, tar_img_name, component_wrapper):
    src_is_color = (src_img_name.split("_")[-1] == "color")
    tar_is_color = (tar_img_name.split("_")[-1] == "color")

    # read image and load component source
    if src_is_color:
        src_img = Image.open(os.path.join(folder_matching, src_img_name))
        src_img = src_img.convert("RGB")
        src_img = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)

        src_mask, src_components = component_wrapper.extract_component(src_img, "color")

    else:
        src_img = Image.open(os.path.join(folder_matching, src_img_name))
        src_img = np.array(src_img)
        src_mask, src_components = component_wrapper.extract_component(src_img, "sketch")

    # read image and load component target
    if tar_is_color:
        tar_img = Image.open(os.path.join(folder_matching, tar_img_name))
        tar_img = tar_img.convert("RGB")
        tar_img = cv2.cvtColor(np.array(tar_img), cv2.COLOR_RGB2BGR)

        tar_mask, tar_components = component_wrapper.extract_component(tar_img, "color")

    else:
        tar_img = Image.open(os.path.join(folder_matching, tar_img_name))
        tar_img = np.array(tar_img)
        tar_mask, tar_components = component_wrapper.extract_component(tar_img, "sketch")

    return src_img, tar_img, src_mask, tar_mask, src_components, tar_components


def process_list_point(list_point):
    list_point = np.array(list_point)
    list_point = list_point[:, :2]
    return list_point

def debug_each_component_v2(src_mask_rgb, tgt_mask_rgb, source_component, target_component, vis_path):
    source_coords = source_component['coords']
    target_coords = target_component['coords']

    src_mask_rgb[source_coords[:,0], source_coords[:,1]] = (255,0,0)
    tgt_mask_rgb[target_coords[:,0], target_coords[:,1]] = (255,0,0)

    dsize = (200, 300)
    debug_images = [cv2.resize(src_mask_rgb, dsize, interpolation=cv2.INTER_NEAREST),
                    cv2.resize(tgt_mask_rgb, dsize, interpolation=cv2.INTER_NEAREST)]

    debug_image = np.concatenate(debug_images, axis=1)
    cv2.imwrite(vis_path, debug_image)

def debug_each_component(src_id, tar_id, src_component, tar_component, src_img, tar_img, vis_folder_pair):
    src_list_points = process_list_point(src_component["coords"])
    tar_list_points = process_list_point(tar_component["coords"])

    src_mask = get_mask(src_list_points, src_img)
    tar_mask = get_mask(tar_list_points, tar_img)

    vis_pair_name = "s" + str(src_id) + "_t" + str(tar_id)

    size = (300, 200)

    src_mask = cv2.merge((src_mask, src_mask, src_mask))
    tar_mask = cv2.merge((tar_mask, tar_mask, tar_mask))
    src_img = cv2.merge((src_img, src_img, src_img))
    tar_img = cv2.merge((tar_img, tar_img, tar_img))

    src_mask = np.array(src_mask, dtype='uint8')
    tar_mask = np.array(tar_mask, dtype='uint8')
    src_img = np.array(src_img, dtype='uint8')
    tar_img = np.array(tar_img, dtype='uint8')

    src_mask = cv2.resize(src_mask, size, interpolation=cv2.INTER_LINEAR)
    tar_mask = cv2.resize(tar_mask, size, interpolation=cv2.INTER_LINEAR)

    src_img = cv2.resize(src_img, size, interpolation=cv2.INTER_LINEAR)
    tar_img = cv2.resize(tar_img, size, interpolation=cv2.INTER_LINEAR)

    vis_mask = cv2.hconcat([src_mask, tar_mask])
    # vis_mask = cv2.merge((vis_mask, vis_mask, vis_mask))

    vis_img = cv2.hconcat([src_img, tar_img])

    vis = np.vstack((vis_mask, vis_img))
    cv2.imwrite(os.path.join(vis_folder_pair, vis_pair_name + ".png"), vis)


def mkdir(path):
    if os.path.isdir(path) == False:
        os.makedirs(path)