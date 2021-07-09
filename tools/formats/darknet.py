import os

from .format import Format
from ..util import write_img


class Darknet(Format):
    def __init__(self, out_folder, image_format):
        self.output_folder = out_folder
        self.image_format = image_format
        self.init_output()

    def init_output(self):
        os.makedirs(self.output_folder+"train/", exist_ok=True)
        os.makedirs(self.output_folder+"test/", exist_ok=True)

    def parse_label(
        self, object_class, img_width, img_height, left_x, bottom_y, right_x, top_y
    ):
        object_width = right_x - left_x
        object_height = top_y - bottom_y
        object_mid_x = (left_x + right_x) / 2.0
        object_mid_y = (bottom_y + top_y) / 2.0

        object_width_rel = object_width / img_width
        object_height_rel = object_height / img_height
        object_mid_x_rel = object_mid_x / img_width
        object_mid_y_rel = object_mid_y / img_height

        dark_net_label = "{} {} {} {} {}".format(
            object_class,
            object_mid_x_rel,
            object_mid_y_rel,
            object_width_rel,
            object_height_rel,
        )

        return dark_net_label
        
    def write_data(self, filename, input_img, input_img_labels, train):
        if train:
            output_file_path = self.output_folder + "train/" + filename
            text_file = open(self.output_folder + "train.txt", "a")
        else:
            output_file_path = self.output_folder + "test/" + filename
            text_file = open(self.output_folder + "test.txt", "a")

        # Save file in general training/testing file
        text_file.write(output_file_path + self.image_format + "\n")
        write_img(output_file_path, self.image_format, input_img)

        # SAVE Darknet TXT FILE WITH THE IMG
        f = open(output_file_path + ".txt", "w+")
        labels_to_print = "\n".join(input_img_labels)
        f.write(labels_to_print)
