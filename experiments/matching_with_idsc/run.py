from datetime import datetime
from utils import ComponentUtils
from experiments.matching_with_idsc._vis_utils import *


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

    component_utils = ComponentUtils(max_contour_points=200)

    #
    source_read_mode = cv2.IMREAD_COLOR if source_mode != 'sketch' else cv2.IMREAD_GRAYSCALE
    source_im = cv2.imread(source_im_path, source_read_mode)

    target_read_mode = cv2.IMREAD_COLOR if target_mode != 'sketch' else cv2.IMREAD_GRAYSCALE
    target_im = cv2.imread(target_im_path, target_read_mode)

    #
    print ('>> on extracting component ...')
    source_mask, source_components = component_utils.extract_component(source_im, source_mode)
    target_mask, target_components = component_utils.extract_component(target_im, target_mode)

    #
    print ('>> on comparing features ...')
    sim_matrix = component_utils.compare_feats(source_components, target_components)
    max_tgt_ids = np.argmax(sim_matrix, 0)

    #
    for c_id, c in enumerate(target_components):
        _path = os.path.join(vis_target_folder, 't%d.png' % c_id)
        cv2.imwrite(_path, c['image'])

    for c_id, c in enumerate(source_components):
        _path = os.path.join(vis_source_folder, 't%d.png' % c_id)
        cv2.imwrite(_path, c['image'])

    for tgt_id, src_id in enumerate(max_tgt_ids):
        print('pair btw src-tgt:', (src_id, tgt_id))
        debug_each_component(src_id, tgt_id, source_components[src_id], target_components[tgt_id],
                             source_im, target_im, vis_matching_folder)



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
        
        # experiments component matching
        print("On experiments matching")
        cost_match = component_wrapper.compare_feats(src_components, tar_components)
        best_match_src_des = np.argmax(cost_match, axis=0) # n_tgt

        vis_folder_pair = os.path.join(vis_folder, img_pair_name)
        mkdir(vis_folder_pair)

        # save component each
        # source
        source_c_folder = os.path.join(vis_folder_pair, 'source')
        mkdir(source_c_folder)

        for c_id, c in enumerate(src_components):
            _path = os.path.join(source_c_folder, 's%d.png' % c_id)
            cv2.imwrite(_path, c['image'])

        # target
        target_c_folder = os.path.join(vis_folder_pair, 'target')
        mkdir(target_c_folder)

        for c_id, c in enumerate(tar_components):
            _path = os.path.join(target_c_folder, 't%d.png' % c_id)
            cv2.imwrite(_path, c['image'])

        for tgt_id, src_id in enumerate(best_match_src_des):
            print ('pair btw src-tgt:', (src_id, tgt_id))
            src_component = src_components[src_id]
            tar_component = tar_components[tgt_id]

            debug_each_component(src_id, tgt_id, src_component, tar_component, src_img, tar_img, vis_folder_pair)

if __name__ == "__main__":
    source_image_path = "../../suneo_image/processed_suneo2.png"
    target_image_path = "../../suneo_image/processed_suneo1.png"

    matching_pair(source_image_path, target_image_path, source_mode='sketch', target_mode='sketch')

    # folder_matching  = "./../../suneo_image"
    # matching(folder_matching)