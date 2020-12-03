from datetime import datetime
import cv2
import time
from utils import ComponentUtils, imgshow
from experiments.matching_with_idsc._vis_utils import *

CONSENSUS = False

def find_target_candidate(source_components, target_components, threshold_distance=50):
    s2t_candidates = {}
    for s_i, s_c in enumerate(source_components):
        s2t_candidates[s_i] = []
        for t_i, t_c in enumerate(target_components):
            # distance
            centroid_distance = t_c['centroid'] - s_c['centroid']
            centroid_distance = np.linalg.norm(centroid_distance, ord=2)

            if centroid_distance < threshold_distance:
                s2t_candidates[s_i] += [t_i]

            #

    return s2t_candidates

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def matching_pair(source_im_path, target_im_path, source_mode, target_mode):
    # initialize modules
    assert source_mode in ['sketch', 'color']
    assert target_mode in ['sketch', 'color']

    img_pair_name = os.path.basename(source_im_path)[:-4] + "-" + os.path.basename(target_im_path)[:-4]
    vis_folder_pair = os.path.join('./output', img_pair_name)
    vis_source_folder = os.path.join(vis_folder_pair, 'source')
    vis_target_folder = os.path.join(vis_folder_pair, 'target')
    vis_matching_folder = os.path.join(vis_folder_pair, 'matching_%s' % datetime.now())

    os.makedirs(vis_source_folder, exist_ok=True)
    os.makedirs(vis_target_folder, exist_ok=True)
    os.makedirs(vis_matching_folder, exist_ok=True)

    component_utils = ComponentUtils(max_contour_points=50, n_angle_bins=9, n_distance_bins=9)

    #
    source_read_mode = cv2.IMREAD_COLOR if source_mode != 'sketch' else cv2.IMREAD_GRAYSCALE
    source_im = cv2.imread(source_im_path, source_read_mode)

    target_read_mode = cv2.IMREAD_COLOR if target_mode != 'sketch' else cv2.IMREAD_GRAYSCALE
    target_im = cv2.imread(target_im_path, target_read_mode)

    #
    print ('>> on extracting component ...')
    source_mask, source_components = component_utils.extract_component(source_im, source_mode)
    target_mask, target_components = component_utils.extract_component(target_im, target_mode)
    source_mask_rgb = cv2.imread(source_im_path)
    target_mask_rgb = cv2.imread(target_im_path)

    print ('source')
    imgshow(source_mask)

    print ('target')
    imgshow(target_mask)

    print ('>> on comparing features ...')
    s_time = time.time()

    dist_mat, matching_info = component_utils.compare_feats(target_components, source_components,
                                                penalty=0.3, min_threshold_area=0.35)
    if CONSENSUS:

        dist_mat_inv, matching_info_inv = component_utils.compare_feats(source_components, target_components,
                                                                        penalty=0.3, min_threshold_area=0.35)
        dist_mat = np.sqrt(dist_mat * dist_mat_inv.T)

    #matching_mat, matching_min, matching_max = matching_info
    max_src_ids = np.argsort(dist_mat, 1) # for each target find the corresponding source
    print ('matching take:', time.time() - s_time)

    #
    t2s_candidates = find_target_candidate(target_components, source_components, threshold_distance=150)
    print ('source_component:', len(source_components))
    print ('target_component:', len(target_components))

    #
    for c_id, c in enumerate(target_components):
        _path = os.path.join(vis_target_folder, 't%d.png' % c_id)
        cv2.imwrite(_path, c['image'])

    for c_id, c in enumerate(source_components):
        _path = os.path.join(vis_source_folder, 's%d.png' % c_id)
        cv2.imwrite(_path, c['image'])

    for tgt_id, src_ids_in_order in enumerate(max_src_ids):
        accepted_src_ids = t2s_candidates[tgt_id]
        src_ids_in_order = [src_id for src_id in src_ids_in_order if src_id in accepted_src_ids]

        if len(src_ids_in_order) > 0:
            src_id = src_ids_in_order[0]
            print('pair btw src-tgt:', (src_id, tgt_id))
            vis_path = os.path.join(vis_matching_folder, 's%d_t%d.png' % (src_id, tgt_id))
            debug_each_component_v2(source_mask_rgb.copy(), target_mask_rgb.copy(), source_components[src_id], target_components[tgt_id], vis_path)

    print ('finish all')

if __name__ == "__main__":
    source_image_path = "../../data/nobita_image/nobita1.png"
    target_image_path = "../../data/nobita_image/nobita2.png"

    #
    matching_pair(source_image_path, target_image_path, source_mode='sketch', target_mode='sketch')