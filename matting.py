from __future__ import division

import numpy as np
import scipy.ndimage
import cv2


TRIMAP_WHITE = 255
TRIMAP_BLACK = 0
TRIMAP_GRAY = 128


def get_border(trimap, area):
    rows, cols = trimap.shape
    return [(i, j) for i in range(1, rows-1) for j in range(1, cols-1)
            if trimap[i, j] == area
            and (trimap[i-1, j] == TRIMAP_GRAY or trimap[i+1, j] == TRIMAP_GRAY
                 or trimap[i, j-1] == TRIMAP_GRAY or trimap[i, j+1] == TRIMAP_GRAY)]


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


def global_matting(img, trimap):
    foreground = trimap == 255
    background = trimap == 0
    unknown = True ^ np.logical_or(foreground, background)
    alpha = foreground

    rows, cols = unknown.shape
    foreborder = get_border(trimap, TRIMAP_WHITE)
    backborder = get_border(trimap, TRIMAP_BLACK)


    return alpha


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

    alpha = global_matting(img, trimap)

    print img[0, 0][0]
    i = [1,2 ,3]
    f = [3, 3,2]
    print [i[x] - f[x] for x in range(len(i))]
    print img[0, 0]
    print img[0,1]
    print img[0, 0] * img[0,1]
    print img[0,0].astype(int)*img[0,1].astype(int)
    print sum(img[0,0].astype(int)) / 3
    show(alpha)


if __name__ == "__main__":
    import scipy.misc
    import matplotlib.pyplot as plt
    main()