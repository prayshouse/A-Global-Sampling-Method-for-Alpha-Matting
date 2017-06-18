from __future__ import division

import numpy as np
import scipy.ndimage
import cv2
import random


TRIMAP_WHITE = 255
TRIMAP_BLACK = 0
TRIMAP_GRAY = 128


# get the border of an area
def get_border(trimap, area):
    rows, cols = trimap.shape
    return [(i, j) for i in range(1, rows-1) for j in range(1, cols-1)
            if trimap[i, j] == area
            and (trimap[i-1, j] == TRIMAP_GRAY or trimap[i+1, j] == TRIMAP_GRAY
                 or trimap[i, j-1] == TRIMAP_GRAY or trimap[i, j+1] == TRIMAP_GRAY)]


# equation (2)
def get_estimate_alpha(i, f, b):
    int_i = i.astype(int)
    int_f = f.astype(int)
    int_b = b.astype(int)
    tmp1 = int_i - int_b
    tmp2 = int_f - int_b
    tmp3 = sum((int_f - int_b) * (int_f - int_b))
    if tmp3 == 0:
        return 0
    else:
        return min(max(tmp1 * tmp2 / tmp3, 0), 1)


# equation (3)
def get_epsilon_c(i, f, b, alpha):
    int_i = i.astype(int)
    int_f = f.astype(int)
    int_b = b.astype(int)
    tmp1 = int_i - (alpha * int_f + (1-alpha) * int_b)
    tmp2 = sum(tmp1 * tmp1)
    return np.sqrt(tmp2)


# equation (4)
def get_epsilon_s(f_x, f_y, i_x, i_y, nearest):
    tmp1 = (f_x - i_x) * (f_x - i_x) + (f_y - i_y) * (f_y - i_y)
    return np.sqrt(tmp1 / nearest)


# equation (5)
def get_epsilon(i, f, b, f_x, f_y, b_x, b_y, i_x, i_y, nearest_f, nearest_b):
    w = 1
    alpha = get_estimate_alpha(i, g, b)
    epsilon_c = get_epsilon_c(i, f, b, alpha)
    epsilon_s_f = get_epsilon_s(f_x, f_y, i_x, i_y, nearest_f)
    epsilon_s_b = get_epsilon_s(b_x, b_y, i_x, i_y, nearest_b)
    return w * epsilon_c + epsilon_s_f + epsilon_s_b


# calculate the nearest distance to border
def get_nearest_distance(x, y, border_list, max_distance_square):
    nearest = max_distance_square
    for i in range(len(border_list)):
        tmp = (border_list[i][0] - x) * (border_list[i][0] - x) + (border_list[i][1] - y) * (border_list[i][1] - y)
        if nearest > tmp:
            nearest = tmp
            nearest_index = i
    return nearest


# match the pixels in unknown areas with a sample
def sample_match(img, trimap, foreborder, backborder, unknown_seq):
    rows, cols, _ = img.shape
    sample = np.zeros((rows, cols, 4), dtype='int')
    for (i, j) in unknown_seq:
        sample[i, j, 0] = random.randint(0, len(foreborder))
        sample[i, j, 1] = random.randint(0, len(backborder))
        sample[i, j, 2] = get_nearest_distance(i, j, foreborder, rows * rows + cols * cols)
        sample[i, j, 3] = get_nearest_distance(i, j, backborder, rows * rows + cols * cols)

    for iteration in range(0, 5):
        # propagation
        for i in range(0, len(unknown_seq)):
            # calculate epsilon among first order neighborhood
            pass


        # Random Search

    return sample


# matting function
def matting(img, trimap):
    print "init params"
    foreground = trimap == 255
    background = trimap == 0
    unknown = True ^ np.logical_or(foreground, background)
    alpha = foreground
    rows, cols = unknown.shape

    print "calculate border"
    foreborder = get_border(trimap, TRIMAP_WHITE)
    backborder = get_border(trimap, TRIMAP_BLACK)
    unknown_seq = [(i, j) for i in range(0, rows) for j in range(0, cols) if trimap[i, j] == TRIMAP_GRAY]

    print "match samples"
    sample = sample_match(img, trimap, foreborder, backborder, unknown_seq)
    print sample[unknown_seq[500][0], unknown_seq[500][1]]
    return alpha


# display the image
def show(img, channel=1):
    if channel == 3:
        plt.imshow(img)
    elif channel == 1:
        plt.imshow(img, cmap='gray')
    else:
        return
    plt.show()


def main():
    print "init"
    img = scipy.misc.imread('troll.png')
    trimap = scipy.misc.imread('trollTrimap.bmp', flatten='True')

    # show(img, 3)
    # show(trimap)

    #alpha = matting(img, trimap)

    # print img[0, 0][0]
    # i = [1,2 ,3]
    # f = [3, 3,2]
    # print [i[x] - f[x] for x in range(len(i))]
    # print img[0, 0]
    # print img[0,1]
    # print img[0, 0] * img[0,1]
    print img[0,0].astype(int)*0.3
    print img[0, 0]
    print img[0, 1]
    print img[0, 2]
    print get_epsilon_c(img[0, 0], img[0, 1], img[0, 0], 0.5)
    # print sum(img[0,0].astype(int)) / 3
    print type(img)
    print img.dtype
    # show(alpha)


if __name__ == "__main__":
    import scipy.misc
    import matplotlib.pyplot as plt
    main()