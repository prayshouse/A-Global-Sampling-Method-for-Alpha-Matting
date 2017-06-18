from __future__ import division

import numpy as np
import scipy.ndimage
import cv2
import random
import sys


TRIMAP_WHITE = 255
TRIMAP_BLACK = 0
TRIMAP_GRAY = 128
FOREBORDER_INDEX = 0
BACKBORDER_INDEX = 1
TO_FOREBORDER_DISTANCE = 2
TO_BACKBORDER_DISTANCE = 3
EPSILON = 4


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
    tmp3 = sum(tmp1 * tmp2)
    tmp4 = sum((int_f - int_b) * (int_f - int_b))
    if tmp4 == 0:
        return 0
    else:
        return min(max(tmp3 / tmp4, 0.0), 1.0)


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
def get_epsilon(i, f, b, f_x, f_y, b_x, b_y, i_x, i_y, f_nearest, b_nearest):
    w = 1
    alpha = get_estimate_alpha(i, f, b)
    epsilon_c = get_epsilon_c(i, f, b, alpha)
    epsilon_s_f = get_epsilon_s(f_x, f_y, i_x, i_y, f_nearest)
    epsilon_s_b = get_epsilon_s(b_x, b_y, i_x, i_y, b_nearest)
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
    sample = np.zeros((rows, cols, 5), dtype='int')
    for (i, j) in unknown_seq:
        sample[i, j, FOREBORDER_INDEX] = random.randint(0, len(foreborder) - 1)
        sample[i, j, BACKBORDER_INDEX] = random.randint(0, len(backborder) - 1)
        sample[i, j, TO_FOREBORDER_DISTANCE] = get_nearest_distance(i, j, foreborder, rows * rows + cols * cols)
        sample[i, j, TO_BACKBORDER_DISTANCE] = get_nearest_distance(i, j, backborder, rows * rows + cols * cols)
        sample[i, j, EPSILON] = sys.maxint

    for iteration in range(0, 10):
        # propagation
        for (m, n) in unknown_seq:
            # calculate epsilon among first order neighborhood
            for (x, y) in [(m, n), (m+1, n), (m-1, n), (m, n+1), (m, n-1)]:
                if 0 <= x < rows and 0 <= y < cols and trimap[x, y] == TRIMAP_GRAY:
                    print x, y
                    f_item = foreborder[sample[x, y, FOREBORDER_INDEX]]
                    b_item = backborder[sample[x, y, BACKBORDER_INDEX]]
                    f = img[f_item[0], f_item[1]]
                    b = img[b_item[0], b_item[1]]
                    i = img[x, y]
                    f_nearest = sample[x, y, TO_FOREBORDER_DISTANCE]
                    b_nearest = sample[x, y, TO_BACKBORDER_DISTANCE]

                    phi = get_epsilon(i, f, b, f_item[0], f_item[1], b_item[0], b_item[1], x, y, f_nearest, b_nearest)
                    if phi < sample[x, y, EPSILON]:
                        sample[m, n, FOREBORDER_INDEX] = sample[x, y, FOREBORDER_INDEX]
                        sample[m, n, BACKBORDER_INDEX] = sample[x, y, FOREBORDER_INDEX]
                        sample[m, n, EPSILON] = phi

        # Random Search
        w = max(len(foreborder), len(backborder))
        for (x, y) in unknown_seq:
            i = img[x, y]
            k = 0
            while True:
                r = w * pow(0.5, k)
                if r < 1:
                    break
                f_new_index = sample[x, y, FOREBORDER_INDEX] + int(r * (random.randint(0, 100 - 1) / 100))
                b_new_index = sample[x, y, BACKBORDER_INDEX] + int(r * (random.randint(0, 100 - 1) / 100))
                if 0 <= f_new_index < len(foreborder) and 0 <= b_new_index < len(backborder):
                    f_item = foreborder[f_new_index]
                    b_item = backborder[b_new_index]
                    f = img[f_item[0], f_item[1]]
                    b = img[b_item[0], b_item[1]]
                    f_nearest = sample[x, y, TO_FOREBORDER_DISTANCE]
                    b_nearest = sample[x, y, TO_BACKBORDER_DISTANCE]
                    phi = get_epsilon(i, f, b, f_item[0], f_item[1], b_item[0], b_item[1], x, y, f_nearest, b_nearest)
                    if phi < sample[x, y, EPSILON]:
                        sample[x, y, FOREBORDER_INDEX] = f_new_index
                        sample[x, y, BACKBORDER_INDEX] = b_new_index
                k = k + 1
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

    print "calculate alpha"
    for x in range(0, rows):
        for y in range(0, cols):
            if trimap[x, y] != TRIMAP_GRAY:
                alpha[x, y] = trimap[x, y]
            else:
                f = img[foreborder[sample[x, y, FOREBORDER_INDEX]][0], foreborder[sample[x, y, FOREBORDER_INDEX]][1]]
                b = img[backborder[sample[x, y, BACKBORDER_INDEX]][0], backborder[sample[x, y, BACKBORDER_INDEX]][1]]
                i = img[x, y]
                alpha[x, y] = get_estimate_alpha(i, f, b)

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

    alpha = matting(img, trimap)
    rows, cols, channel = img.shape
    show(alpha)
    count = 0
    for i in range(0, rows):
        for j in range(0, cols):
            if 0 < alpha[i, j] < 1:
                count = count + 1
    print "count", count

    plt.imshow((alpha.reshape(rows, cols, 1).repeat(3, 2)*img).astype(np.uint8))
    plt.show()

if __name__ == "__main__":
    import scipy.misc
    import matplotlib.pyplot as plt
    main()