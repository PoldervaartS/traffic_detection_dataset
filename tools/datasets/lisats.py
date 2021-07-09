from sklearn.model_selection import train_test_split
import os
import shutil
import zipfile
import logging
from tqdm import tqdm
import pandas as pd

from ..common import Common
from ..util import read_img, get_img_dim
from .dataset import Dataset


# All dataset parsers inheret from Dataset.
class LISA_TS(Dataset):
    def __init__(self, lisats_config: dict, config: Common):
        """
        lisats_config: dict
            Dict of values loaded from the dataset.yaml file
            expected keys:
            [RAW_ROOT]      -> root path of raw dataset.zips
            [ZIP]           -> name of zip file, with .zip
            [CLEAN_ROOT]    -> where to extract the cleaned files.
            [PREPEND]       -> prefix to all .txt and .jpg files outputted

        config: Common
            A Common object created from loaded yaml files with the configs.
        """

        # path to raw file (absolute)
        self.raw_path = config.root + lisats_config["RAW_FOLDER"] + lisats_config["ZIP"]

        # path to initialization folder (absolute)
        self.init_path = config.init_folder + lisats_config["INIT_FOLDER"]
        self.annot_file = self.init_path + "allAnnotations.csv"

        self.prepend = lisats_config["PREPEND"]
        self.config = config

    # ===============================================
    # initializes dataset into intermediary unzipped and cleaned state.
    def init_dataset(self):
        """
        unzips lisats into intermidiary INIT_FOLDER
        cleans up files from unzip to minimize disk space.
        """
        assert os.path.isfile(self.raw_path)  # makes sure that the zip actually exists.
        print("Started initializing LISATS!")

        # make subfolder inside INIT_FOLDER path.
        os.makedirs(self.init_path, exist_ok=True)

        # unzips file into directory
        with zipfile.ZipFile(self.raw_path, "r") as zip_ref:
            # zip_ref.extractall(self.init_path)
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                zip_ref.extract(member, self.init_path)

        # delete some of the unnecessary folders.
        delete_folders = ["negatives/", "tools/"]
        for folder in delete_folders:
            shutil.rmtree(self.init_path + folder)

        # modify allAnnotations.csv -> replace semicolons with commas
        with open(self.annot_file, "r") as file:
            filedata = file.read()
        filedata = filedata.replace(";", ",")
        with open(self.annot_file, "w") as file:
            file.write(filedata)

        print("Finished initializing LISATS!")

    # ================================================
    # parses dataset into output folder

    def parse_label(self, img_annotations, img_width, img_height):
        labels = []
        for _, row in img_annotations.iterrows():
            img_class = self.config.classes_dict[row["Annotation"]]

            x_left = float(row["Upper left corner X"])
            x_right = float(row["Lower right corner X"])

            # don't ask me why, bit its reversed. Authors never mentioned this
            y_lower = float(row["Upper left corner Y"])
            y_upper = float(row["Lower right corner Y"])

            # error checking. I am cranky
            assert y_lower < y_upper
            assert x_left < x_right

            labels.append(
                self.config.formatter.parse_label(
                    img_class,
                    img_width,
                    img_height,
                    x_left,
                    y_lower,
                    x_right,
                    y_upper,
                )
            )

        return labels


    def parse_dataset(self):
        assert os.path.isfile(self.annot_file)
        print("Started Parsing LISA_TS")
        annot_df = pd.read_csv(self.annot_file, sep=",", header=0, usecols=list(range(6)))

        # 1. cleans up labels. see simplify_class functions
        def simplify_class(raw_annotation):
            """
            takes raw annotations, and makes it simple.
            Removes numbers, so all speed limit signs are defined as speed limit.
            remove unrdbl from annotations.

            thruTrafficMergeLeft == thruMergeLeft, so all of the former is changed to the latter.
            """
            clean_annotation = ""
            if "Urdbl" in raw_annotation:
                clean_annotation = raw_annotation[:-5]
            elif raw_annotation[-2].isnumeric():
                clean_annotation = raw_annotation[:-2]
            else:
                clean_annotation = raw_annotation

            # dataset authors didn't notice that they had the same sign labeled differently.
            if clean_annotation == "thruTrafficMergeLeft":
                clean_annotation = "thruMergeLeft"
            return clean_annotation

        annot_df["Annotation"] = annot_df.apply(
            lambda row: simplify_class(row["Annotation tag"]), axis=1
        )

        # 2. Prune signs that won't be relevant in competition.
        signs_to_drop = [
            "truckSpeedLimit",
            "schoolSpeedLimit",
            "rampSpeedAdvisory",
            "zoneAhead",
            "dip",
            "addedLane",
        ]
        annot_df.drop(
            annot_df[annot_df["Annotation"].isin(signs_to_drop)].index,
            axis=0,
            inplace=True,
        )

        # 3. Grabs list of unique image files
        images = list(annot_df["Filename"].unique())

        # 4. make training and testing split
        img_train, img_test = train_test_split(
            images, 
            test_size=self.config.test_split,
            random_state=self.config.random_state
        )

        # 5. generate the labels for each image, and write it out.
        for img_file in tqdm(images):
            image = read_img(self.init_path + img_file, self.config.color_mode)
            img_width, img_height = get_img_dim(image)

            # Grab all labels for this image.
            img_annotations = annot_df[annot_df["Filename"] == img_file]
            
            # convert class/label to numerical for darknet
            labels = self.parse_label(img_annotations, img_width, img_height)

            # write out labels and image
            filename = self.prepend + img_file.split("/")[-1]
            self.config.formatter.write_data(filename, image, labels, (img_file in img_train))
        # end for loop

        print("Finished parsing LISA_TS")
