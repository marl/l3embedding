import numpy as np
import skimage.color

def adjust_saturation(rgb_img, factor):
    hsv_img = skimage.color.rgb2hsv(rgb_img)
    hsv_img[:,:,1] = np.clip(hsv_img[:,:,1] * factor, 0.0, 1.0)
    return skimage.color.hsv2rgb(hsv_img)


def adjust_brightness(rgb_img, delta):
    return np.clip(rgb_img - delta, 0.0, 1.0)

def horiz_flip(rgb_img):
    return rgb_img[:,::-1,:]
