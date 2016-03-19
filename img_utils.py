from np_array_utils import *


def apply_mask_to_img(base_img, mask):
    base_img = fromarray(base_img)
    base_img.paste(mask, (0, 0), mask)
    return toarray(base_img)