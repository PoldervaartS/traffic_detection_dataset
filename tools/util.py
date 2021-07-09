import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
    



# Reads the image using opencv.
def read_img(input_img_file_path, color_mode):
    return cv2.imread(input_img_file_path, color_mode)

# Returns the image width and height for opencv image.
def get_img_dim(input_img):
    img_height, img_width, _ = input_img.shape
    return img_width, img_height

# Write an opencv image
def write_img(output_file_path, image_format,output_img):
    return cv2.imwrite(output_file_path + image_format, output_img)

# takes in the image width and height, as well as any darknet labels
# and outlines it on the image
# TODO: slow class along with box outline.
def show_img(img, labels, image_width, image_height):
    _, ax = plt.subplots(1)
    ax.imshow(img)

    for raw_label in labels:
        # convert back to matplotlib format
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
        label_items = raw_label.split(" ")
        box_width = image_width * float(label_items[3])
        box_height = image_height * float(label_items[4])
        x = (image_width * float(label_items[1])) - (box_width / 2)
        y = (image_height * float(label_items[2])) - (box_height / 2)

        rect = patches.Rectangle(
            (x, y),
            box_width,
            box_height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()