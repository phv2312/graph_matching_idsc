import numpy as np
import os 
import sys 
import cv2
sys.path.append("../")
import utils
from itertools import combinations
from utils import ComponentUtils
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

def debug_each_component(src_id, tar_id, src_component, tar_component, src_img, tar_img, vis_folder_pair):

    src_list_points = process_list_point(src_component["coords"])
    tar_list_points = process_list_point(tar_component["coords"])

    src_mask = get_mask(src_list_points, src_img)
    tar_mask = get_mask(tar_list_points, tar_img)

    vis_pair_name = "s"+str(src_id)+"_t"+str(tar_id)


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
    cv2.imwrite(os.path.join(vis_folder_pair, vis_pair_name+".png"), vis)

def mkdir(path):
    if os.path.isdir(path) == False:
        os.makedirs(path)


def matching(folder_matching):
    vis_folder = "./result_matching"
    list_file_name = os.listdir(folder_matching)

    comb_list = build_binary_combination(list_file_name)
    for each_pair in comb_list:
        src_img_name = each_pair[0]
        tar_img_name = each_pair[1]

        img_pair_name = src_img_name[:-4]+"-"+tar_img_name[:-4]
        print("Matching pair:" + img_pair_name)
        
        # extract component
        print("On extracting component")
        component_wrapper = ComponentUtils(10)
        src_img, tar_img, src_mask, tar_mask, src_components, tar_components = component_extract(folder_matching, src_img_name, tar_img_name, component_wrapper)
        
        # debug component matching
        print("On debug matching")
        cost_match = component_wrapper.compare_feats(src_components, tar_components)
        best_match_src_des = np.argmin(cost_match, axis=1)

        vis_folder_pair = os.path.join(vis_folder, img_pair_name)
        mkdir(vis_folder_pair)
        for src_id, src_component in enumerate(src_components):
            tar_id = best_match_src_des[src_id]
            tar_component = tar_components[tar_id]

            debug_each_component(src_id, tar_id, src_component, tar_component, src_img, tar_img, vis_folder_pair)

if __name__ == "__main__":
    folder_matching  = "../suneo_image"
    matching(folder_matching)