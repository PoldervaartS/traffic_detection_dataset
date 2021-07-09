from sklearn.model_selection import train_test_split
import os
import shutil
import zipfile
import logging
import pandas as pd
import glob
from tqdm import tqdm
from pathlib import Path

from ..common import Common
from ..util import read_img, get_img_dim
from .dataset import Dataset


# All dataset parsers inheret from Dataset.
class LISA_TL(Dataset):
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
        unzips lisatl into intermidiary INIT_FOLDER
        cleans up files from unzip to minimize disk space.
        """
        
        assert os.path.isfile(self.raw_path)  # makes sure that the zip actually exists.
        print("Started initializing LISATL!")

        # make subfolder inside INIT_FOLDER path.
        os.makedirs(self.init_path, exist_ok=True)

        # unzips file into directory
        with zipfile.ZipFile(self.raw_path, "r") as zip_ref:
            # zip_ref.extractall(self.init_path)
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                zip_ref.extract(member, self.init_path)

        
        # rename nightTrain, dayTrain to nightTraining, dayTraining (match annotation file)
        shutil.move(self.init_path+"nightTrain/nightTrain", self.init_path+"nightTraining")
        shutil.move(self.init_path+"dayTrain/dayTrain", self.init_path+"dayTraining")

        # delete some of the unnecessary folders.
        delete_folders = ["sample-nightClip1/", "sample-dayClip6/", "nightSequence1", "nightSequence2", "daySequence1", "daySequence2", "nightTrain", "dayTrain"]
        for folder in delete_folders:
            shutil.rmtree(self.init_path + folder)


        # read in all day annotations
        total_day_df = []
        for dayClip in tqdm([x for x in Path(self.init_path+"Annotations/Annotations/dayTrain/").glob('**/*') if x.is_dir()]):
            path = os.path.join(dayClip, "frameAnnotationsBOX.csv")
            total_day_df.append(pd.read_csv(path, sep=";"))

        tdf_day = pd.concat(total_day_df)
        tdf_day["day"] = 1

        # read in all night annotations
        total_night_df = []
        for nightClip in tqdm([x for x in Path(self.init_path+"Annotations/Annotations/nightTrain/").glob('**/*') if x.is_dir()]):
            path = os.path.join(nightClip, "frameAnnotationsBOX.csv")
            total_night_df.append(pd.read_csv(path, sep=";"))

        tdf_night = pd.concat(total_night_df)
        tdf_night["day"] = 0


        all_annotations_df = pd.concat([tdf_day, tdf_night]) # combine the two annotations

        # fixes filepaths
        all_annotations_df.rename(columns={"Filename": "Filename_old"}, inplace=True)
        def fix_filename(origin_col, file_col):
            path = origin_col.split("/")[:2]        # adds the first two folders ie dayTraining, dayClip1
            path.append("frames")                   # adds the frames folder
            path.append(file_col.split("/")[-1])    # adds the image file.
            return "/".join(path)
        all_annotations_df["Filename"] = all_annotations_df.apply(
            lambda row: fix_filename(row["Origin file"], row["Filename_old"]), axis=1
        )


        all_annotations_df.to_csv(self.annot_file) # output to csv

        print("Finished initializing LISATL!")

    # ================================================
    # parses dataset into output folder

    def parse_label(self, img_annotations, img_width, img_height):
        labels = []
        for _, row in img_annotations.iterrows():
            img_class = self.config.classes_dict[row["Annotation"]]

            x_left = float(row["Upper left corner X"])
            x_right = float(row["Lower right corner X"])
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
        print("Started Parsing LISA_TL")
        wanted_cols = ["Filename", "Annotation tag", "Upper left corner X", "Lower right corner X", "Upper left corner Y", "Lower right corner Y"]
        annot_df = pd.read_csv(self.annot_file, sep=",", header=0, usecols=wanted_cols)


        # 1. cleans up labels. see simplify_class functions
        def simplify_class(curr_annotation):
        # currently a placeholder for fancier data manipulation.
            return "trafficLight"

        annot_df["Annotation"] = annot_df.apply(
            lambda row: simplify_class(row["Annotation tag"]), axis=1
        )


        # 2. Grabs list of unique image files
        images = list(annot_df["Filename"].unique())

        # 3. make training and testing split
        img_train, img_test = train_test_split(
            images, 
            test_size=self.config.test_split,
            random_state=self.config.random_state
        )

        # 4. generate the labels for each image, and write it out.
        for img_file in tqdm(images):
            assert os.path.isfile(self.init_path + img_file)

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

        print("Finished parsing LISA_TL")
