from scipy.sparse.csgraph import floyd_warshall
from skimage.draw import line as skline
import numpy as np
import cv2
import scipy as sp, scipy.spatial
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
        for k in range(nh1.shape[0]):
            if nh1[k] + nh2[k] == 0:
                continue
            cost += (nh1[k] - nh2[k]) ** 2 / (nh1[k] + nh2[k])
        return cost / 2.0

    n1, n2 = len(histo1), len(histo2)
    dists = np.zeros(shape=(n1, n2), dtype=np.float32)

    for i1 in range(n1):
        for i2 in range(n2):
            dists[i1, i2] = shape_context_cost(histo1[i1], histo2[i2])

    return dists

#############

"""
DPMatching Utils
"""
def road_to_pair(road):
    h, w = road.shape[:2]
    pair = []

    i, j = h-1, w-1
    while(i >= 0 and j>=0):
        direction = road[i,j]
        ofs_i, ofs_j = 0, 0
        if direction == 1.:
            pair += [(i,j)]
            ofs_i, ofs_j = -1, -1
        elif direction == 2.:
            ofs_i, ofs_j = 0, -1
        elif direction == 0:
            ofs_i, ofs_j = -1, 0

        i += ofs_i
        j += ofs_j

    # the first pair is assumed to be matched correctly.
    return [(0,0)] + pair

def dp_matching(d):
    h, w = d.shape
    g = np.zeros((h, w))
    road = np.copy(g)
    g[0, 0] = d[0, 0]

    for t in range(1, w):
        g[0, t] = g[0, t - 1] + d[0, t]
    for n in range(1, h):
        g[n, 0] = g[n - 1, 0] + d[n, 0]

    for i in range(1, h):
        for j in range(1, w):
            temp = [g[i - 1, j] + d[i, j], g[i - 1, j - 1] + d[i, j] * 2, g[i, j - 1] + d[i, j]]
            step = int(np.argmin(np.array(temp)))
            number = temp[step]
            g[i, j] = number
            road[i, j] = step

    score = float(g[h - 1, w - 1])
    return score, road_to_pair(road)

"""
Main Discriptor
"""
class IDSCDescriptor:
    def __init__(self, n_contour_points=2, n_angle_bins=8,
                 n_distance_bins=8):
        self.n_contour_points = n_contour_points
        self.n_angle_bins = n_angle_bins
        self.n_distance_bins = n_distance_bins

        self.distance = sp.spatial.distance.euclidean
        self.shortest_path = floyd_warshall

    def describe(self, binary):
        print("Here", binary.shape)
        self.max_distance = self.distance((0, 0), binary.shape)
        contour_points = self._sample_contour_points(binary, self.n_contour_points)

        if len(contour_points) == 0:
            print('contours missing in IDSC')
            return np.zeros(self.n_contour_points)

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
        idx = np.linspace(0, len(contour_points) - 1, num=n).astype(np.int)
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

def matching(histo1, histo2, contours1, contours2):
    n1, n2      = len(histo1), len(histo2)
    distance    = calc_cost(histo1, histo2)

    scores, pair_ids, perm2s = [], [], []
    first_ids2 = np.argsort(distance[0])[:4]
    distance_res = []
    for first_id2 in first_ids2:
        perm2   = list(range(first_id2, n2)) + list(range(0, first_id2))
        perm2   = np.array(perm2)
        histo2_ = histo2[perm2]

        distance_           = calc_cost(histo1, histo2_)
        score, pair_ids_    = dp_matching(distance_)
        distance_res.append(distance_)

        #
        scores      += [score]
        pair_ids    += [pair_ids_]
        perm2s      += [perm2]

    min_id2     = int(np.argmin(scores))
    perm2       = perm2s[min_id2]
    distance_get = distance_res[min_id2]
    contours2   = contours2[perm2]
    histo2      = histo2[perm2]
    pair_ids    = pair_ids[min_id2]

    # Note that histo2 & contours2 may be different from the initialize.
    return pair_ids, histo1, histo2, contours1, contours2, distance_get

if __name__ == '__main__':
    im_path1 = "/home/kan/Desktop/cinnamon/active_learning/data/1a.png"
    im_path2 = "/home/kan/Desktop/cinnamon/active_learning/data/1b.png"

    im1 = cv2.imread(im_path1)
    im2 = cv2.imread(im_path2)
    im_bin1 = to_binary(im_path1)
    im_bin2 = to_binary(im_path2)
    idsc_descriptor = IDSCDescriptor(n_contour_points=100,
                                     n_angle_bins=8,
                                     n_distance_bins=8)

    histo1, contours1 = idsc_descriptor.describe(im_bin1)
    histo2, contours2 = idsc_descriptor.describe(im_bin2)

    #
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    max_h = max(h1, h2)
    if max_h == h1:
        im2 = cv2.copyMakeBorder(im2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif max_h == h2:
        im1 = cv2.copyMakeBorder(im1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    debug_im = np.concatenate([im1, im2], axis=1)
    offset_x = w1

    pair_ids, histo1, histo2, contours1, contours2, _ = matching(histo1, histo2, contours1, contours2)
    for (i1, i2) in pair_ids:
        _debug_im = debug_im.copy()

        p = contours1[i1]
        q = contours2[i2]

        xp, yp = p
        xq, yq = q

        cv2.circle(_debug_im, (xp, yp), radius=2, color=(255, 0, 0), thickness=2)
        cv2.circle(_debug_im, (offset_x + xq, yq), radius=2, color=(255, 0, 0), thickness=2)
        cv2.line(_debug_im, (xp, yp), (offset_x + xq, yq), color=(0, 255, 0), thickness=1)

        plt.imshow(_debug_im)
        plt.show()