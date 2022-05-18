##############################################################################
# FILE: cartoonify_old_V.py
# WRITERS: noam shabat
# EXERCISE: Intro2cs2 ex6 2021-2022
# DESCRIPTION: this project is about creating a new filter to a picture in a
#              sens of cartoon to it.
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
import copy
import math
import sys
from typing import Optional

import ex6_helper as h
from ex6_helper import *

"""
Magic Variables:
"""
GRAY_RED = 0.299

GRAY_GREEN = 0.587

GRAY_BLUE = 0.114
"""
-_-_-_ End _-_-_-
"""


def separate_channels(image: h.ColoredImage) -> List[List[List[int]]]:
    """
    reorder of the image data in to channels X rows X columns.
    :param image:3D list that contains an image data
    :return: separate_channels
    """
    columns = len(image[0])
    rows = len(image)
    channels = len(image[0][0])
    final_list = []
    two_dim = []  # an empty two multidimensional list.
    one_dim = []  # an empty one multidimensional list.
    for dim in range(channels):
        for row in range(rows):
            for col in range(columns):
                one_dim.append(image[row][col][dim])
            two_dim.append(one_dim)
            one_dim = []
        final_list.append(two_dim)
        two_dim = []
    return final_list


def combine_channels(channels: List[List[List[int]]]) -> ColoredImage:
    """
    this func is undoing the separate_channels function.
    :param channels: 3D list that contains an image data
    :return: the previous organization of the data rows X columns X channels.
    """
    return separate_channels(separate_channels(channels))


def grey_pixel(pixel):
    """
    this func calculates the black & white color of the pixel in the image.
    :param pixel: the pixel that his color is being change.
    :return:the Black & White color of the change.
    """
    temp_val = round(
        pixel[0] * GRAY_RED + pixel[1] * GRAY_GREEN + pixel[2] * GRAY_BLUE)
    if temp_val > 255:
        temp_val = 255
    if temp_val < 0:
        temp_val = 0
    return temp_val


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """
    the func is changing gets a colored image the returns a Black & White
    colored image.
    :param colored_image: get an image that is in color
    :return: Black & White colored image.
    """
    grey_image = [[] for _ in range(len(colored_image))]
    for index_row in range(len(colored_image)):
        for index_col in range(len(colored_image[index_row])):
            grey_pix = grey_pixel(colored_image[index_row][index_col])
            grey_image[index_row].append(grey_pix)
    return grey_image


def blur_kernel(size: int) -> Kernel:
    """
    a kernel that implement a blur in to the image.
    :param size: the size of the kernel.
    :return: a blur to the image.
    """
    # Set "size" to be its abs value to deal with negative values
    size = abs(size)
    blur_ls_new = []
    blur = 1 / (size ** 2)
    for row in range(size):
        blur_temp_row = []
        for val in range(size):
            blur_temp_row.append(blur)
        blur_ls_new.append(blur_temp_row)

    return blur_ls_new


def validate_index(row, col, image_mat):
    """
    checks if the coordinates are in the image.
    :param row: the length of the row.
    :param col: the length of the col.
    :param image_mat: 2D list that contains an image data.
    :return: True if the val have the right parameters.
    """
    if row >= len(image_mat) or row < 0:
        return False
    if col >= len(image_mat[0]) or col < 0:
        return False
    return True


def get_env(index_row, index_col, env_size, image_mat):
    """
    in this func we are creating the environment of the pixel that
    is being altered.
    :param index_row: the index of a value in the row.
    :param index_col: the index of a value in the columns.
    :param env_size: the size of the pixel we are looking in to.
    :param image_mat: 2D list that contains an image data.
    :return:  the environment of the pixel we what to change.
    """
    env_pixel = []
    delta = (env_size - 1) / 2
    for row in range(env_size):
        temp_row_ls = []
        for index in range(env_size):
            index_x = int(index_row - delta + row)
            index_y = int(index_col - delta + index)
            if validate_index(index_x, index_y, image_mat):
                temp_row_ls.append(image_mat[index_x][index_y])
            else:
                temp_row_ls.append(image_mat[index_row][index_col])
        env_pixel.append(temp_row_ls)
    return env_pixel


def kernel_calc(env, kernel):
    """
    the func is calculating the kernel effect on the image.
    :param env: the endearment of the kernel.
    :param kernel: a matrix that have an effect when is being implemented.
    :return:the sum of the expected change of the kernel.
    """
    sum_env = 0
    for row in range(len(env)):
        for col in range(len(env[0])):
            sum_env += env[row][col] * kernel[row][col]
    if sum_env < 0:
        sum_env = 0
    if sum_env > 255:
        sum_env = 255
    return round(sum_env)


def apply_kernel(image: SingleChannelImage,
                 kernel: Kernel) -> SingleChannelImage:
    """
    in this func we are implementing the kernel in to the image to do the
    changes that we want it to do.
    :param image: in image presented in a form of 3D list that contains
    the image data.
    :param kernel: the matrix that is being implemented on the pixel's
    in the image.
    :return: the changed image ofter the kernel effect.
    """
    blur_image = []
    for row_index in range(len(image)):
        temp_row = []
        for col_index in range(len(image[row_index])):
            env = get_env(row_index, col_index, len(kernel), image)
            pix_val = kernel_calc(env, kernel)
            temp_row.append(pix_val)
        blur_image.append(temp_row)
    return blur_image


def get_pixel_corners_val(image, y, x):
    """
    the func is calculating the value of the four cords that create a
    square around the pixel that we are checking.
    :param image: 2D list that contains an image data
    :param x: coordinates of the x val for the pixel.
    :param y: coordinates of the y val for the pixel.
    :return: the coordinates of the four cords that create a square around the
    pixel.
    """

    ceil_x = math.ceil(x)
    floor_x = math.floor(x)
    ceil_y = math.ceil(y)
    floor_y = math.floor(y)
    # a
    top_left = image[floor_y][floor_x]
    # b
    bottom_left = image[ceil_y][floor_x] if ceil_y < len(image) else 0
    # c
    top_right = image[floor_y][ceil_x] if ceil_x < len(image[0]) else 0
    # d
    bottom_right = image[ceil_y][ceil_x] if ceil_x < len(
        image[0]) and ceil_y < len(image) else 0

    return top_right, top_left, bottom_right, bottom_left


def bilinear_interpolation(image: SingleChannelImage, y: float,
                           x: float) -> int:
    """
    calculate the pixel i the Surrounding it pixel in the image.
    :param image: 2D list that contains an image data
    :param x: coordinates of the x val for the pixel.
    :param y: coordinates of the y val for the pixel.
    :return: the calculation regarding the pixel and the distances from the
    edge's from the four pixels that Surrounding him.
    """
    top_right, top_left, bottom_right, bottom_left = get_pixel_corners_val(
        image, y, x)
    x = x - math.floor(x)
    y = y - math.floor(y)
    calc_for_interpolation = top_left * (1 - x) * (1 - y) + bottom_left * y * \
                             (1 - x) + top_right * x * (
                                     1 - y) + bottom_right * x * y
    return round(calc_for_interpolation)


def resize(image: SingleChannelImage, new_height: int,
           new_width: int) -> SingleChannelImage:
    """
    the func is resizes the image into the desirable sizes.
    :param image: 2D list that contains an image data
    :param new_height:
    :param new_width:
    :return: the new desirable size for the image.
    """
    new_image = []
    # Turns out ration should be calculated on "max index" rather than size
    height_ratio = (len(image) - 1) / (new_height - 1)
    width_ratio = (len(image[0]) - 1) / (new_width - 1)
    for row in range(new_height):
        new_row = []
        for col in range(new_width):
            pixel_val = bilinear_interpolation(image, row * height_ratio,
                                               col * width_ratio)
            new_row.append(pixel_val)
        new_image.append(new_row)
    return new_image


def scale_down_colored_image(image: ColoredImage, max_size: int) -> Optional[
    ColoredImage]:
    """
    this func is will check if an image data stands with if the requirements
    :param image:  2D list that contains an image data.
    :param max_size: number of the max size.
    :return: if it stands in the requirements return none else return a new image.
    """
    if len(image) <= max_size and len(image[0]) <= max_size:
        return None

    ratio = max_size / max(len(image), len(image[0]))
    new_image_row_len = round(len(image) * ratio)
    new_image_col_len = round(len(image[0]) * ratio)

    im = separate_channels(image)

    channel_1 = resize(im[0], new_image_row_len, new_image_col_len)
    channel_2 = resize(im[1], new_image_row_len, new_image_col_len)
    channel_3 = resize(im[2], new_image_row_len, new_image_col_len)

    return combine_channels([channel_1, channel_2,
                             channel_3])


def rotate_90(image: Image, direction: str) -> Image:
    """
    rotate the image and changes the order of the pixel in 2D list.
    :param image: a 2D list that contains an image data
    :param direction: the direction of to rotate.
    :return: a rotation of the image or to the right or to the left.
    """
    if direction == "R":
        rev_lst_r = []
        for i in range(len(image[0])):
            res = []
            for j in range(len(image)):
                res.append(image[-1 - j][i])
            rev_lst_r.append(res)
        return rev_lst_r

    if direction == "L":
        rev_lst_l = []
        for i in range(len(image[0])):
            res = []
            for j in range(len(image)):
                res.append(image[j][-1 - i])
            rev_lst_l.append(res)
        return rev_lst_l


def avg_env(env):
    """
    this func is giving back the average sum from the env of the pixels.
    :param env: the env of the pixel.
    :return: the average sum from the env of the pixels.
    """
    sum_env = 0
    num_of_pixels = 0
    for i in range(len(env)):
        for j in range(len(env[0])):
            sum_env += env[i][j]
            num_of_pixels += 1
    return sum_env / num_of_pixels


def sum_for_get_edges(env):
    """
    sum of the edges in the Surrounding pixels.
    :param env: the env of the pixel.
    :return: the sum of the edges.
    """
    sum_env = 0
    for row in range(len(env)):
        for col in range(len(env[0])):
            sum_env += env[row][col]
    return sum_env


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int,
              c: int) -> SingleChannelImage:
    """
    this func is getting the edges from the original image.
    :param image:a 2D list that contains an image data
    :param blur_size:a temporary version for the image.
    :param block_size: the size of the
    :param c: a parameter that we are
    :return:an image with the same dim and with only two colors.(B & W)
    """
    blur_image = apply_kernel(image, blur_kernel(blur_size))
    new_image = []
    for i in range(len(image)):
        row_im = []
        for j in range(len(image[0])):
            # changed integer division to regular division
            threshold = sum_for_get_edges(
                get_env(i, j, block_size, blur_image)) / (block_size ** 2)
            if blur_image[i][j] < threshold - c:
                row_im.append(0)
            else:
                row_im.append(255)
        new_image.append(row_im)
    return new_image


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    """
    this func is calculating from a given formula.
    :param image: a 2D list that contains an image data.
    :param N: num of channels.
    :return: a calc from a given formula.
    """
    quantized_image = copy.deepcopy(image)
    for k, i in enumerate(image):
        for j, l in enumerate(i):
            p_image = round(math.floor(image[k][j] * (N / 256)) * (255 / (N - 1)))
            quantized_image[k][j] = p_image
    return quantized_image


def quantize_colored_image(image: ColoredImage, N: int) -> ColoredImage:
    """
    this func is calculating from a given formula.
    :param image:a 3D list that contains an image data.
    :param N:num of channels.
    :return:an image calculated from a given formula.
    """
    res = []
    sep = separate_channels(image)
    for i in sep:
        k = quantize(i, N)
        res.append(k)
    res = combine_channels(res)
    return res



def add_mask_two_or_more_channels(image1, image2, mask):
    """
    combine two images into one and every image have two colors or more.
    :param image1: a 3D list that contains an image1 data
    :param image2: a 3D list that contains an image2 data
    :param mask: the action that combine the two pic's.
    :return: the new image after the change.
    """
    new_image = []
    for j, image in enumerate(image1):
        new_image.append([])
        for chan in range(len(image)):
            new_image[j].append([])
    for c, col in enumerate(image1):
        for pixel, channel in enumerate(col):
            for p, k in enumerate(channel):
                new_image[c][pixel].append(round(
                    k * mask[c][pixel] + image2[c][pixel][p] * (
                            1 - mask[c][pixel])))
    return new_image


def add_mask_one_channel(image1, image2, mask):
    """
    combine two images into one and every image have two colors or more.
    :param image1: a 3D list that contains an image1 data
    :param image2: a 3D list that contains an image2 data
    :param mask: the action that combine the two pic's.
    :return: the new image after the change.
    """
    new_image = []
    for i, x in enumerate(image1):
        new_image.append([])
    for i, z in enumerate(image1):
        for pixel, channel in enumerate(z):
            new_image[i].append(round(
                channel * mask[i][pixel] + image2[i][pixel] * (
                        1 - mask[i][pixel])))
    return new_image


def add_mask(image1: Image, image2: Image, mask: List[List[float]]) -> Image:
    """
    the func make sure how many color's there is in the image's and combine
    two images into one.
    :param image1: a 3D list that contains an image1 data
    :param image2: a 3D list that contains an image2 data
    :param mask: the action that combine the two pic's.
    :return:the new image after the change.
    """
    if type(image1[0][0]) == list:
        new_image = add_mask_two_or_more_channels(image1, image2, mask)

    else:
        new_image = add_mask_one_channel(image1, image2, mask)

    return new_image


def cartoonify(image: ColoredImage, blur_size: int, th_block_size: int,
               th_c: int, quant_num_shades: int) -> ColoredImage:
    """
        this func is combining the hole program together into the sole purpose
        to cartoonify an image.
        :param image: image:3D list that contains an image data
        :param blur_size: the size of the blur.
        :param th_block_size:the size of the block size for the desired
        image parameters.
        :param th_c: a num that is being subtracted from the get edges func.
        :param quant_num_shades:num of channels.
        :return: a different image from the original after have been cartoonify.
        """
    step_1 = quantize_colored_image(image, quant_num_shades)
    step_2 = RGB2grayscale(image)
    step_3 = get_edges(step_2, blur_size, th_block_size, th_c)
    for i, l in enumerate(step_3):
        for n in range(len(l)):
            step_3[i][n] = (step_3[i][n] // 255)
    new_temp_list = []
    for i, k in enumerate(step_1):
        new_temp_list.append([])
        for z, p in enumerate(k):
            new_temp_list[i].append([])
            for q in p:
                new_temp_list[i][z].append(0)
    cartoon_image_final = add_mask(step_1, new_temp_list, step_3)
    return cartoon_image_final


def main():
    """
    this func is initializing the program.
    :return: None
    """
    if len(sys.argv) == 8:
        # args
        image_source = sys.argv[1]
        cartoon_dest = sys.argv[2]
        max_im_size = int(sys.argv[3])
        blur_size = int(sys.argv[4])
        th_block_size = int(sys.argv[5])
        th_c = int(sys.argv[6])
        quant_num_shades = int(sys.argv[7])

        # load image
        image = h.load_image(image_source)

        # resize if needed
        if len(image) > max_im_size or len(image[0]) > max_im_size:
            image = scale_down_colored_image(image, max_im_size)

        image = cartoonify(image, blur_size, th_block_size, th_c,
                           quant_num_shades)

        # save
        h.save_image(image, cartoon_dest)
    else:
        print("error : invalid number of argument was inputted")


if __name__ == '__main__':
    # global image
    main()
