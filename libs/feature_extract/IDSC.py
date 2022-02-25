import os
from scipy.sparse.csgraph import floyd_warshall
from skimage.draw import line as skline
import numpy as np
import cv2
import scipy as sp, scipy.spatial
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

"""
Image Utils
"""


def to_binary(im, thresh=128):
    if type(im) is str:
        im = cv2.imread(im, 0)

    _, bin = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    return bin

#############


"""
Cost Utils
"""


def calc_cost(histo1, histo2):
    def shape_context_cost(nh1, nh2):
        '''
            nh1, nh2 -> histogram
            return cost of shape context of
            two given shape context of the shape.
        '''
        cost = 0
        if nh1.shape[0] > nh2.shape[0]:
            nh1, nh2 = nh2, nh1
        nh1 = np.hstack([nh1, np.zeros(nh2.shape[0] - nh1.shape[0])])

        sub_ = np.power(nh1 - nh2, 2.)
        add_ = nh1 + nh2

        nonzero_ids = np.nonzero(add_)[0]
        cost = sub_[nonzero_ids] / add_[nonzero_ids]

        return 0.5 * np.sqrt(np.sum(cost))

    n1, n2 = len(histo1), len(histo2)
    dists = cdist(histo1, histo2, metric=lambda e1, e2: shape_context_cost(e1, e2))

    return dists

#############

"""
DPMatching Utils
"""


def dp_matching(A, penalty):
    def MAT_GET(pMat, x, y, nRow):
        return pMat[y,x]

    def MAT_SET(pMat, x, y, val, nRow):
        pMat[y,x] = val

    uplimit = 100000000
    M, N = A.shape[:2]
    D = np.zeros(shape=(M, N), dtype=np.float32)
    links = np.ones(shape=(M, N), dtype=np.int32)
    C = np.zeros(shape=(M,), dtype=np.int32)

    dTmp = A[0, 0]
    if dTmp < penalty:
        D[0, 0] = dTmp
        links[0, 0] = 1
    else:
        D[0, 0] = penalty
        links[0, 0] = 2

    for pt2 in range(1, N):
        dTmp1 = MAT_GET(A, pt2, 0, M) + pt2 * penalty
        dTmp3 = MAT_GET(D, pt2-1,0,M) + penalty
        if dTmp1 < dTmp3:
            MAT_SET(D, pt2, 0, dTmp1, M)
            MAT_SET(links, pt2, 0, 1, M)
        else:
            MAT_SET(D, pt2, 0, dTmp3, M)
            MAT_SET(links, pt2, 0, 3, M)

    for pt1 in range(1, M):
        dTmp1 = MAT_GET(A, 0, pt1, M) + pt1 * penalty
        dTmp2 = MAT_GET(D, 0, pt1 - 1, M) + penalty
        if dTmp1 < dTmp2:
            MAT_SET(D, 0, pt1, dTmp1, M)
            MAT_SET(links, 0, pt1, 1, M)
        else:
            MAT_SET(D, 0, pt1, dTmp2, M)
            MAT_SET(links, 0, pt1, 2, M)

    for pt1 in range(1, M):
        for pt2 in range(1, N):
            dTmp1 = MAT_GET(D, pt2 - 1, pt1 - 1, M) + MAT_GET(A, pt2, pt1, M)
            dTmp2 = MAT_GET(D, pt2, pt1 - 1, M) + penalty
            dTmp3 = MAT_GET(D, pt2 - 1, pt1, M) + penalty

            if dTmp1<=dTmp2 and dTmp1<=dTmp3:
                MAT_SET(D, pt2, pt1, dTmp1, M)
                MAT_SET(links, pt2, pt1, 1, M)
            elif dTmp2<=dTmp3:
                MAT_SET(D, pt2, pt1, dTmp2, M)
                MAT_SET(links, pt2, pt1, 2, M)
            else:
                MAT_SET(D, pt2, pt1, dTmp3, M)
                MAT_SET(links, pt2, pt1, 3, M)

    OCL = 4 * N
    C_dct = []
    for pt1 in range(0, M):
        C[pt1] = OCL

    T = MAT_GET(D, N-1, M -1, M)
    if T < uplimit:
        pt1 = M - 1
        pt2 = N - 1
        while (pt1 >= 0 and pt2 >= 0):
            link = MAT_GET(links, pt2, pt1, M)
            if link == 1:
                C[pt1] = pt2
                C_dct += [(pt1, pt2)]
                pt1 -= 1
                pt2 -= 1
            elif link == 2:
                pt1 -= 1
            elif link == 3:
                pt2 -= 1
            else:
                print ("links[pt1,pt2]=%d, FAILED!!" % MAT_GET(links,pt2,pt1,M))
                return False
    else:
        print ("Terminate without computing C,  T=%lf", T)
        return False

    return C, C_dct, T


def multi_dp_matching(A, penalty, n_start, n_search):
    def ROUND(x):
        return int(x + 0.5)

    M, N = A.shape
    A2 = np.zeros(shape=(M,2*N), dtype=np.float32)
    A2[:, :N] = A
    A2[:, N:] = A
    bSucc = False

    CC = np.zeros(shape=(N,M), dtype=np.int32)
    id_best = -1
    T_best = 4000 * N
    n_start = min(N, n_start)

    for iS in range(0, n_start):
        id_start = ROUND(N*iS/float(n_start))
        C, _, TT = dp_matching(A2[:, id_start:id_start+N], penalty)
        CC[id_start] = C.copy()
        if TT < T_best:
            T_best = TT
            id_best = id_start

        for dS in range(1, n_search):
            # forward
            iS1 = id_start + dS
            if iS1>N: iS1 -= N
            C, _, TT = dp_matching(A2[:, iS1:iS1+N], penalty)
            CC[iS1] = C.copy()
            if TT < T_best:
                T_best = TT
                id_best = iS1

            # backward
            iS2 = id_start - dS
            if iS2<0: iS2 += N
            C, _, TT = dp_matching(A2[:, iS2:iS2+N], penalty)
            CC[iS2] = C.copy()
            if TT < T_best:
                T_best = TT
                id_best = iS2

    if id_best < 0:
        print ("\n\tId_best=%d !!!!!!!\n" % id_best)
        return False

    C = CC[id_best]
    for ii in range(0, M):
        if C[ii] < N:
            C[ii] += id_best
            if (C[ii] >= N): C[ii] -= N

    return C, T_best

"""
Main Discriptor
"""


class IDSCDescriptor:
    def __init__(self, max_contour_points=100, n_angle_bins=8, n_distance_bins=8):
        self.max_contour_points = max_contour_points
        self.n_angle_bins = n_angle_bins
        self.n_distance_bins = n_distance_bins

        self.distance = sp.spatial.distance.euclidean
        self.shortest_path = floyd_warshall

        self.max_distance = None

    def describe(self, binary, given_contour_points=None):
        self.max_distance = self.distance((0, 0), binary.shape)
        if given_contour_points is None:
            contour_points = self._sample_contour_points(binary, self.max_contour_points)
        else:
            contour_points = given_contour_points

        if len(contour_points) == 0:
            print('contours missing in IDSC')
            return np.zeros(self.max_contour_points)

        dist_matrix = self._build_distance_matrix(binary, contour_points)
        context = self._build_shape_context(dist_matrix, contour_points)

        return context, contour_points

    def _get_points_on_line(self, p1, p2):
        x, y = skline(p1[0], p1[1], p2[0], p2[1])
        return x, y

    def _sample_contour_points(self, binary, n):
        ct, img = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_points = max(ct, key=len)
        # remove second dim
        contour_points = np.reshape(contour_points, (len(contour_points), 2))
        # sample n points
        num_samples = self.max_contour_points# fixed here
        idx = np.linspace(0, len(contour_points) - 1, num=num_samples).astype(np.int)

        return contour_points[idx]

    def _build_distance_matrix(self, binary, contour_points):
        dist_matrix = np.zeros((len(contour_points), len(contour_points)))

        # fill the distance matrix pairwise
        for i, p1 in enumerate(contour_points):
            for j, p2 in enumerate(contour_points[i + 1:]):
                lx, ly = self._get_points_on_line(p1, p2)
                values = binary[ly, lx]
                inside_shape = np.count_nonzero(values) == len(values)
                if not inside_shape:
                    continue
                # if all points on line are within shape -> calculate distance
                dist = self.distance(p1, p2)
                if dist > self.max_distance:
                    break
                # store distance in matrix (mirrored)
                dist_matrix[j + i, i] = dist
                dist_matrix[i, j + i] = dist

        return dist_matrix

    def _build_shape_context(self, distance_matrix, contour_points, skip_distant_points=False):
        histogram = []
        max_log_distance = np.log2(self.max_distance)
        # steps between assigned bins
        dist_step = max_log_distance / self.n_distance_bins
        angle_step = np.pi * 2 / self.n_angle_bins
        # find shortest paths in distance matrix (distances as weights)
        graph = self.shortest_path(distance_matrix, directed=False)

        # iterate all points on contour
        for i, (x0, y0) in enumerate(contour_points):
            hist = np.zeros((self.n_angle_bins, self.n_distance_bins))

            # calc. contour tangent from previous to next point
            # to determine angles to all other contour points
            (prev_x, prev_y) = contour_points[i - 1]
            (next_x, next_y) = contour_points[(i + 1) % len(contour_points)]
            tangent = np.arctan2(next_y - prev_y, next_x - prev_x)

            # inspect relationship to all other points (except itself)
            # direction and distance are logarithmic partitioned into n bins
            for j, (x1, y1) in enumerate(contour_points):
                if j == i:
                    continue
                dist = graph[i, j]
                # 0 or infinity determine, that there is no path to point
                if dist != 0 and dist != np.inf:
                    log_dist = np.log2(dist)
                # ignore unreachable points, if requested
                elif skip_distant_points:
                    continue
                # else unreachable point is put in last dist. bin
                else:
                    log_dist = max_log_distance
                angle = (tangent - np.arctan2(y1 - y0, x1 - x0)) % (2 * np.pi)
                # calculate bins, the inspected point belongs to
                dist_idx = int(min(np.floor(log_dist / dist_step), self.n_distance_bins - 1))
                angle_idx = int(min(angle / angle_step, self.n_angle_bins - 1))
                # point fits into bin
                hist[angle_idx, dist_idx] += 1

            # L1 norm
            if hist.sum() > 0:
                hist = hist / hist.sum()
            histogram.append(hist.flatten())

        return np.array(histogram)


def matching(histo1, histo2, penalty=0.3, n_start=4, n_search=2, max_thresh_dist=np.inf):
    distance = calc_cost(histo1, histo2)
    C, score = multi_dp_matching(distance, n_start=n_start, n_search=n_search, penalty=penalty)
    pair_ids = C2pair(C)
    pair_ids = [(i1, i2) for i1, i2 in pair_ids if distance[i1, i2] < max_thresh_dist]

    return pair_ids, distance, score


def C2pair(C, unmatch_num=200):
    C = np.array(C)

    good_idxs = np.where(C != unmatch_num)[0]
    source_idxs = np.arange(C.shape[0])[good_idxs]
    target_idxs = C[good_idxs]

    return np.stack([source_idxs, target_idxs], axis=1)


if __name__ == '__main__':
    im_path1 = "_"
    im_path2 = "_"

    print (os.path.basename(im_path1), os.path.basename(im_path2))

    im1 = cv2.imread(im_path1)
    im2 = cv2.imread(im_path2)
    im_bin1 = to_binary(im_path1)
    im_bin2 = to_binary(im_path2)

    idsc_descriptor = IDSCDescriptor(max_contour_points=50,
                                     n_angle_bins=8,
                                     n_distance_bins=8)

    histo1, contours1 = idsc_descriptor.describe(im_bin1)
    histo2, contours2 = idsc_descriptor.describe(im_bin2)
    s2t_pair, distance, score = matching(histo1, histo2,
                                         penalty=0.3, max_thresh_dist=0.45,
                                         n_start=4, n_search=2)
    print ('matching score:', score)

    # visualize
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    max_h = max(h1, h2)
    if max_h == h1:
        im2 = cv2.copyMakeBorder(im2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif max_h == h2:
        im1 = cv2.copyMakeBorder(im1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    debug_im = np.concatenate([im1, im2], axis=1)
    offset_x = w1

    for (i1, i2) in s2t_pair:
        _debug_im = debug_im.copy()
        _d = distance[i1, i2]

        p = contours1[i1]
        q = contours2[i2]

        xp, yp = p
        xq, yq = q

        cv2.circle(_debug_im, (xp, yp), radius=2, color=(255, 0, 0), thickness=2)
        cv2.circle(_debug_im, (offset_x + xq, yq), radius=2, color=(255, 0, 0), thickness=2)
        cv2.line(_debug_im, (xp, yp), (offset_x + xq, yq), color=(0, 255, 0), thickness=1)

        print ('dist btw %d-%d is %.3f:', (i1, i2, _d))
        plt.imshow(_debug_im)
        plt.show()