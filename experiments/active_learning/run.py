import os
import cv2
import numpy as np
import random
from utils import ALUtils, ComponentUtils, imgshow

# global for short
component_utils = ComponentUtils(max_contour_points=100, n_angle_bins=8, n_distance_bins=8)
al_utils = ALUtils()

def resize_image(mask, im_h, im_w):
    return cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

def get_dummy_label(reference_sketch_path, target_sketch_path):
    r_bn = os.path.basename(reference_sketch_path)
    t_bn = os.path.basename(target_sketch_path)

    if (r_bn, t_bn) in [('processed_suneo2.png', 'processed_suneo1.png')]:
        pair =[(1,1),(2,2),(5,3),(3,3),(4,2),(6,8),(7,8),(8,9),(9,9),(10,13),(13,17),(11,17),(15,18),
               (14,18),(17,23),(16,22),(19,20),(18,24)] # target-reference in order
        return pair

    raise Exception('not having dummy label for pair:', (r_bn, t_bn))

def active_learning(r_infos, t_infos, r_labels_tuple):
    r_mask, r_components = r_infos
    t_mask, t_components = t_infos
    r_labels_dict  = {_1:_2 for _1,_2 in r_labels_tuple}
    r_labels_tuple = np.array(r_labels_tuple)

    n_r = len(r_components)
    n_t = len(t_components)
    n_lbl = np.max(r_labels_tuple[:, 1]) + 1

    # build one-hot reference label
    reference_labels = np.zeros(shape=(n_r + n_t, n_lbl), dtype=np.float32)
    observed = np.zeros(shape=(n_r + n_t,), dtype=np.bool)
    for r_i, r_c in enumerate(r_components):
        r_mask = r_c['mask']
        if r_mask in r_labels_dict:
            reference_labels[r_i, r_labels_dict[r_mask]] = 1.
            observed[r_i] = True
        else:
            observed[r_i] = False

    #
    all_components = r_components + t_components
    if not os.path.exists('dummy_k.npy'):
        K = component_utils.compare_feats_al(all_components, penalty=0.3, alpha=15.)
        np.save('dummy_k.npy', K)
    else:
        K = np.load('dummy_k.npy')

    print ('K shape:', K.shape)

    #
    label_hat = al_utils.estimate_label(K, reference_labels, observed)
    print ('label_hat shape:', label_hat.shape)

    # predict
    non_observed_idxs = np.where(observed == False)[0]
    predict_ids = np.argmax(label_hat, axis=1) # (n_t, n_lbl)

    assert len(non_observed_idxs) == len(predict_ids)

    print ('--> new')

    for _i in range(len(predict_ids)):

        pass

    for t_id, lbl_id in enumerate(predict_ids):

        print ('predict component-mask-%d with lbl-%d' % (target_components[t_id], lbl_id))

def get_feature(c):
    feat, sampling_points = c['idsc']
    return np.concatenate([feat, sampling_points], axis=1)

from scipy.spatial.distance import cdist
def simple_kernel(components):
    feats = [c['idsc'][0].flatten() for c in components]
    feats = np.array(feats)

    K = cdist(feats, feats, metric='cosine')
    K = np.exp(-0.25 * K)

    return K

from scipy.sparse import csgraph
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
def solve_semi(K, y_train):
    """
    K: large is best
    please set y_train[i] = -1 if it's unlabelled.
    """
    n_samples = K.shape[0]

    laplacian = csgraph.laplacian(K, normed=True)
    laplacian = -laplacian

    if sparse.isspmatrix(laplacian):
        diag_mask = (laplacian.row == laplacian.col)
        laplacian.data[diag_mask] = 0.0
    else:
        laplacian.flat[::n_samples + 1] = 0.0

    graph_matrix = laplacian

    classes = np.unique(y_train)
    classes = (classes[classes != -1])
    classes_ = classes

    n_samples, n_classes = len(y_train), len(classes)
    alpha = 0.2

    y = np.asarray(y_train)
    unlabeled = y == -1

    label_distributions_ = np.zeros((n_samples, n_classes))
    for label in classes:
        label_distributions_[y == label, classes == label] = 1

    y_static = np.copy(label_distributions_)
    y_static *= 1 - alpha

    l_previous = np.zeros((n_samples, n_classes))
    unlabeled = unlabeled[:, np.newaxis]
    if sparse.isspmatrix(graph_matrix):
        graph_matrix = graph_matrix.tocsr()

    max_iter = 100
    tol = 1e-3
    for n_iter_ in range(max_iter):
        if np.abs(label_distributions_ - l_previous).sum() < tol:
            break

        l_previous = label_distributions_
        label_distributions_ = safe_sparse_dot(
            graph_matrix, label_distributions_)

        label_distributions_ = np.multiply(
            alpha, label_distributions_) + y_static

    normalizer = np.sum(label_distributions_, axis=1)[:, np.newaxis]
    normalizer[normalizer == 0] = 1
    label_distributions_ /= normalizer

    # set the transduction item
    transduction = classes_[np.argmax(label_distributions_,
                                           axis=1)]
    transduction_ = transduction.ravel()

    return transduction_, classes_

def calc_custom_K(X1, X2):

    pass

def add_flatten_feat(r_components):
    for c in r_components:
        c['idsc_flatten'] = c['idsc'][0].flatten()

def main():
    r_sketch_path = "../../data/nobita_image/nobita1.png"
    t_sketch_path = "../../data/nobita_image/nobita2.png"
    r_sketch = cv2.imread(r_sketch_path, 0)
    t_sketch = cv2.imread(t_sketch_path, 0)

    print ('on extracting component ...')
    r_mask, r_components = component_utils.extract_component(r_sketch, mode='sketch')
    t_mask, t_components = component_utils.extract_component(t_sketch, mode='sketch')
    add_flatten_feat(r_components)
    add_flatten_feat(t_components)
    imgshow(r_mask)
    imgshow(t_mask)

    # dummy label, for this specified reference & target only
    lbl_names = ['toc', 'tran', 'kinh', 'mieng', 'ao', 'no', 'tay', 'quan', 'dui_chan', 'do_chan']
    lbl_ids = [0,1,2,3,4,5,6,7,8,9]
    r_m2lblid = {
        1:0, 2:1, 3:2, 4:2, 5:3, 7:4, 8:4, 6:5, 10:6, 11:6, 9:7, 12:8, 13:8, 14:9, 15:9
    }

    #
    a_components = r_components + t_components
    X = np.vstack([c['idsc_flatten'] for c in a_components])
    y = np.ones(shape=(len(a_components))) * -1
    y[np.arange(len(r_components))] = [r_m2lblid.get(c['mask'],-1) for c in r_components]

    from sklearn.semi_supervised import LabelSpreading
    model = LabelSpreading(gamma=0.25,max_iter=200)
    model.fit(X,y)

    print('predict:\n', {m: l for l, m in zip(model.transduction_[len(r_components):], [c['mask'] for c in t_components])})
    imgshow(t_mask)
    exit()

    #
    print ('on calculating K')
    gamma = 10.

    if not os.path.exists('K.npy'):
        K, _ = component_utils.compare_feats(a_components, a_components, penalty=.3, min_threshold_area=.35)

        np.save('K.npy', K)
        print ('load from scratch')
    else:
        K = np.load('K.npy')
        print ('load from file')

    K = np.sqrt(K * K.T)
    K = np.exp(-1 / gamma * K)

    #
    print ('on solving ')
    y_hat, _ = solve_semi(K, y)
    print ('predict:\n', {m:l for l, m in zip(y_hat[len(r_components):], [c['mask'] for c in t_components])})
    imgshow(t_mask)

if __name__ == '__main__':
    #
    main()
    #
    # #
    # reference_sketch_path = "../../data/suneo_image/processed_suneo2.png"  # reference manually
    # target_sketch_path = "../../data/suneo_image/processed_suneo1.png"  # target manually
    # new_sketch_path = "../../data/suneo_image/processed_suneo3.png"  # entiryle new image
    # DEBUG = False
    #
    # #
    # reference_sketch = cv2.imread(reference_sketch_path, 0)
    # target_sketch = cv2.imread(target_sketch_path, 0)
    # new_sketch = cv2.imread(new_sketch_path, 0)
    #
    # reference_mask ,reference_components = component_utils.extract_component(reference_sketch, mode='sketch')
    # target_mask, target_components = component_utils.extract_component(target_sketch, mode='sketch')
    # #new_mask, new_components = component_utils.extract_component(new_sketch, mode='sketch')
    #
    # #imgshow(target_mask)
    #
    # if DEBUG:
    #     rh, rw = reference_mask.shape[:2]
    #     th, tw = target_mask.shape[:2]
    #
    #     mh = max(rh, th)
    #     mw = max(rw, tw)
    #     reference_mask_vis = resize_image(reference_mask, mh, mw)
    #     target_mask_vis = resize_image(target_mask, mh, mw)
    #     _mask_vis = np.concatenate([reference_mask_vis, target_mask_vis], axis=1)
    #     imgshow(_mask_vis)
    #
    # # building id2mask
    # n_t = len(target_components)
    # n_r = len(reference_components)
    # target_id2mask = {i: c['mask'] for i, c in enumerate(target_components)}
    # target_mask2id = {c: i for i, c in target_id2mask.items()}
    #
    # #
    # human_label = get_dummy_label(reference_sketch_path, target_sketch_path) # list of tuple (target_id, reference_id)
    #
    # labels = [l[1] for l in human_label]
    # id2label = {i:l for i,l in enumerate(labels)}
    # label2id = {l:i for i,l in id2label.items()}
    #
    # human_label = [(target_mask2id[_1], label2id[_2]) for _1, _2 in human_label]
    # human_label_dict = {_1:_2 for _1,_2 in human_label}
    # #human_label = np.array(human_label)
    # y = np.array([human_label_dict.get(i, -1) for i in range(n_t)])
    #
    # random_removed_ids = [target_mask2id[mask] for mask in [13, 15, 19]]
    # y_ = y.copy()
    # y[random_removed_ids] = -1
    #
    # if not os.path.exists('K.npy'):
    #     K = component_utils.compare_feats_consensus(target_components, target_components) #np.exp(-0.05 * K) #np.exp(-0.25 * K)
    #     np.save('K.npy', K)
    # else:
    #     K = np.load('K.npy', allow_pickle=True)
    #
    # K = K[0] #np.exp(-.25 * (1 - K[0]))
    # print (K.flatten())
    #
    # transduction, n_classes = solve_semi(K, y)
    # print (random_removed_ids)
    # print ([id2label[p] for p in transduction[random_removed_ids]])
    # print ([(_1, _2) for (_1, _2) in zip(transduction, y_)])
    #
    # exit()
    #
    # # building one-hot label
    # one_hot_labels = np.zeros(shape=(n_t, n_lbl), dtype=np.float32)
    # observed = np.zeros(shape=(n_t), dtype=np.float32)
    #
    # for t_i in range(n_t):
    #     t_mask = target_id2mask[t_i]
    #
    #     if t_mask in human_label_dict:
    #         one_hot_labels[t_i, human_label_dict[t_mask]] = 1.
    #         observed[t_i] = 1.
    #     else:
    #         observed[t_i] = 0.
    #
    # observed[random_removed_ids] = 0.
    # observed = observed.astype(np.bool)
    # one_hot_labels[random_removed_ids, :] = 0.
    #
    # if not os.path.exists('_dummy_k.npy'):
    #     K = component_utils.compare_feats_al(target_components, penalty=0.3, alpha=15.)
    #     np.save('_dummy_k.npy', K)
    # else:
    #     K = np.load('_dummy_k.npy')
    #
    # print ('K shape:', K.shape)
    #
    # label_hats = al_utils.estimate_label(K, labels=one_hot_labels, observed=observed)
    #
    # # training
    # print ('training')
    # observed_ids = np.where(observed == True)[0]
    # for t_id, one_hot_label in enumerate(one_hot_labels):
    #     if np.sum(one_hot_label) == 0:
    #         continue
    #
    #     t_mask = target_id2mask[t_id]
    #     label_hat_mask = np.argmax(one_hot_label)
    #     print('training, target-mask: %d with reference-mask: %d' % (t_mask, label_hat_mask))
    #
    # # predict
    # print ('predicting ...')
    # non_observed_ids = np.where(observed == False)[0]
    # assert label_hats.shape[0] == non_observed_ids.shape[0]
    # for non_observed_id, label_hat in zip(non_observed_ids, label_hats):
    #     target_mask = target_id2mask[non_observed_id]
    #     label_hat_mask = np.argmax(label_hat)
    #     K_ = K[non_observed_id]
    #     K_[non_observed_id] = 10.
    #     best_neighbor_target_mask = target_id2mask[np.argmin(K_)]
    #
    #     print ('predicting, target-mask: %d with reference-mask: %d' % (target_mask, label_hat_mask))
    #     print ('predicting, the best neighbor:', best_neighbor_target_mask)
    #
    # #
    # # #
    # # active_learning(r_infos=(target_mask, target_components), t_infos=(new_mask, new_components),
    # #                 r_labels_tuple=human_label)
    # #
    # # print('finish')
