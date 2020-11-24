import os
import cv2
import numpy as np
from utils import ALUtils, ComponentUtils, imgshow

#
al_utils = ALUtils()
component_utils = ComponentUtils(dist_per_point=15)

#
image_dir = "data/suneo_image"
sketch_names = ["processed_suneo2.png", "processed_suneo1.png"]
color_name   = "processed_suneo2_color.png"

#
sketch_paths = [os.path.join(image_dir, s_name) for s_name in sketch_names]
color_path = os.path.join(image_dir, color_name)

color_image  = cv2.imread(color_path)

sketch_images = [cv2.imread(sketch_path, 0) for sketch_path in sketch_paths]
sketch_image = sketch_images[0]
other_sketch_images = sketch_images[1:]

#
sketch_info = component_utils.extract_component(sketch_image, mode='sketch')
other_sketch_infos = [component_utils.extract_component(s, mode='sketch') for s in other_sketch_images]
color_info   = component_utils.extract_component(color_image, mode='color')

print ('-- visualize sketch mask')
#imgshow(sketch_info[0])

print ('-- visualize color mask')
#imgshow(color_info[0])

if True:
    source_info = sketch_info # label
    target_info = other_sketch_infos[0] # un-label

    source_mask, source_component = source_info
    target_mask, target_component = target_info

    imgshow(source_mask)
    imgshow(target_mask)

    imgshow(source_component[3]['image'])
    imgshow(target_component[16]['image'])

    cv2.imwrite("data/1a.png", source_component[3]['image'])
    cv2.imwrite("data/16b.png", target_component[16]['image'])
    exit()

    imgshow(source_component[1]['image'])
    imgshow(target_component[4]['image'])
    component_utils._compare_feat_single(source_component[1]['idsc'], target_component[1]['idsc'], threshold=0.2)
    component_utils._compare_feat_single(source_component[1]['idsc'], target_component[4]['idsc'], threshold=0.2)


    imgshow(source_mask)
    imgshow(target_mask)

    # prepare one hot
    n_src = len(source_component)
    n_tgt = len(target_component)

    print ('n_src:', n_src)
    print ('n_tgt:', n_tgt)

    label = np.zeros(shape=(n_src + n_tgt, n_src), dtype=np.float32)
    observed = np.zeros(shape=(n_src + n_tgt,), dtype=np.bool)
    for s_i, s_c in enumerate(source_component):
        label[s_i][s_i] = 1.
        observed[s_i] = True

    all_component = source_component + target_component
    K_ = component_utils.compare_feats(source_component, target_component, threshold=0.4)
    K = component_utils.compare_feats(all_component, all_component, threshold=0.2)
    print ('K shape:', K.shape)

    label_hat = al_utils.estimate_label(K, labels=label, observed=observed)
    print ('label hat:', label_hat, label_hat.shape)

    predict_ids = np.argmin(label_hat, axis=-1)
    for t_id, s_id in enumerate(predict_ids):
        print('pair btw t/s:', (t_id, s_id))
        print('pair btw t/s:', (t_id, np.argsort(label_hat[t_id])[::-1]))