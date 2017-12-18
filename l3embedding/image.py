import numpy as np
import skimage
import skimage.color

def adjust_saturation(rgb_img, factor):
    """
    Adjust the saturation of an RGB image

    Args:
        rgb_img: RGB image data array
        factor: Multiplicative scaling factor to be applied to saturation

    Returns:
        adjusted_img: RGB image with adjusted saturation
    """
    hsv_img = skimage.color.rgb2hsv(rgb_img)
    imin, imax = skimage.dtype_limits(hsv_img)
    hsv_img[:,:,1] = np.clip(hsv_img[:,:,1] * factor, imin, imax)
    return skimage.color.hsv2rgb(hsv_img)


def adjust_brightness(rgb_img, delta):
    """
    Adjust the brightness of an RGB image

    Args:
        rgb_img: RGB image data array
        delta: Additive (normalized) gain factor applied to each pixel

    Returns:
        adjusted_img: RGB image with adjusted saturation
    """
    imin, imax = skimage.dtype_limits(rgb_img)
    # Convert delta into the range of the image data
    delta = rgb_img.dtype.type((imax - imin) * delta)

    return np.clip(rgb_img + delta, imin, imax)

def horiz_flip(rgb_img):
    """
    Horizontally flip the given image

    Args:
        rgb_img: RGB image data array

    Returns:
        flipped_img: Horizontally flipped image
    """
    return rgb_img[:,::-1,:]
