from sklearn.model_selection import train_test_split
import shutil
import zipfile
import logging
from tqdm import tqdm
from pathlib import Path
from ..common import Common
from ..util import read_img, get_img_dim
from .dataset import Dataset
import glob, os
import pathlib
import shutil
import pandas as pd
import numpy as np
import time
import cv2
import sys
from PIL import Image as Img

class bdd100kTL(Dataset):
    def __init__(self, bdd100k_config: dict, config: Common):
        """
        bdd100k_config: dict
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
        self.raw_path = config.root + bdd100k_config["RAW_FOLDER"] + bdd100k_config["ZIP"]
        self.annot_file = config.root + bdd100k_config["RAW_FOLDER"] + bdd100k_config["ANNOTATION_ZIP"]
        # path to initialization folder (absolute)
        self.init_path = config.init_folder + bdd100k_config["INIT_FOLDER"]
 
        self.prepend = bdd100k_config["PREPEND"]
        self.config = config

    # ===============================================
    # initializes dataset into intermediary unzipped and cleaned state.
    
    def unzip_func(self, raw_path):
        
        assert os.path.isfile(raw_path)  # makes sure that the zip actually exists.
        # make subfolder inside INIT_FOLDER path.        
        
        os.makedirs(self.init_path, exist_ok=True)
                    
        # unzips file into directory
        with zipfile.ZipFile(raw_path, "r") as zip_ref:
            # zip_ref.extractall(self.init_path)
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                zip_ref.extract(member, self.init_path)   

           
    def init_dataset(self):
        """
        unzips lisatl into intermidiary INIT_FOLDER
        cleans up files from unzip to minimize disk space.
        """
        
        assert os.path.isfile(self.raw_path)  # makes sure that the zip actually exists.
        print("Started initializing bdd100kTL!")

        # List of the zipped files. 
        # raw_path => image files, annot_file => labels (annotation file)
        
        #*               Unzip files                       *#
        
        zip_list = [self.raw_path, self.annot_file]
                        
        for i in range (len(zip_list)):
            self.unzip_func(self, zip_list[i])
        
        #*               Unzip files End                   *#
        
        crr_path = os.path.abspath(os.getcwd()) 
        
        annotation_path = crr_path + "/InitDataset/bdd100k/bdd100k/labels/"
        
        image_path = crr_path + "/InitDataset/bdd100k/bdd100k/images/100k/"
        
        # Initialize variables
        x_Right = 0; x_Left = 0; y_Upper = 0; y_Bottom = 0; key= ""; label_in_dictionary = False;

        jpg_name = ""; category = ""; trafficLightColor=""
        
        # Prefix is used to distinguish val / train files
        prefix = ""
        
        # new_row is used to add it to a dataframe
        new_row = "" 
        
        # Creating a folder for TL. This will only contain image files that have traffic lights
        try:
            os.mkdir(crr_path + "/InitDataset/bdd100k/bdd_TL")
        except OSError as error:
            print(error)
            
        # Store paths
        annotation_path = crr_path + "/InitDataset/bdd100k/bdd100k/labels/"        
        image_path_100k = crr_path + "/InitDataset/bdd100k/bdd100k/images/100k/"            
        image_path = crr_path + "/InitDataset/bdd100k/bdd100k/images"
        csv_path = crr_path + "/InitDataset/bdd100k/"
        #*           Start creating annotation file          *#
        
        for infiles in sorted(glob.glob(annotation_path + '*.json')):
            
            # Create dataframes for traffic Light annotation files        
            df_trafficLight = pd.DataFrame(columns =['Filename', 'Annotation tag', 'x Left', 'y Upper', 'x Right',
                              'y Bottom'] )
                        
            if "val" in infiles:
                prefix = "val"
            if "train" in infiles:
                prefix = "train"
            
            with open (infiles, 'r') as myfile:
                for myline in myfile:
                    
                    # Extract file name
                    if "name" in myline:
                        jpg_name = myline.split("\"")[3]
                    
                    # Extract category. Category contains "traffic light"
                    # We are only going to use "traffic light" 
                    if "category" in myline:
                        category = myline.split("\"")[3]
                    
                    # Extract x1, x2, y1, y2 values
                    if category == "traffic light": 
                        if "x1" in myline:
                            x_Left = myline.split("\"")[2]
                            x_Left = x_Left.replace(',', '')
                            x_Left = x_Left.replace(':', '')
                        if "x2" in myline:                    
                            if "box2d" not in myline:
                                x_Right = myline.split("\"")[2]
                                x_Right = x_Right.replace(',', '')
                                x_Right = x_Right.replace(':', '')
                        if "y1" in myline:
                            y_Bottom = myline.split("\"")[2]
                            y_Bottom = y_Bottom.replace(',', '')
                            y_Bottom = y_Bottom.replace(':', '')
                        if "y2" in myline:
                            y_Upper = myline.split("\"")[2] 
                            y_Upper = y_Upper.replace(',', '')
                            y_Upper = y_Upper.replace(':', '')      
                            
                            # After we get y2, new jpg or category starts
                            # So we create a new row and add it to the dataframe
                            
                            # Creating a new row that contains filename, annotation tag, x, y values
                            if category == "traffic light":
                                if (trafficLightColor != 'none'):                                    
                                    new_row = {'Filename': jpg_name, 'Annotation tag': "trafficLight",
                                                'Annotation LightTag': "trafficLight" + trafficLightColor,
                                                'x Left' :x_Left, 'y Upper' :y_Upper, 'x Right':x_Right,
                                                'y Bottom' : y_Bottom}
                                    df_trafficLight = df_trafficLight.append(new_row, ignore_index = True)

                                category = "";
                            y_Upper = 0
                            
                        if category == "traffic light":
                                if "trafficLightColor" in myline:
                                    trafficLightColor = myline.split("\"")[3]
            
            # Create annotation.csv for traffic light
            df_trafficLight.to_csv(crr_path + "/InitDataset/bdd100k/" + prefix + "_df_trafficLight.csv")
            df_TL_no_dup = df_trafficLight.drop_duplicates(subset=['Filename'])

            # Copy necessary files and move them to bdd_TL folder. 
            
            for fileName in df_TL_no_dup['Filename']:
                try:
                    shutil.copy(image_path_100k + prefix + '/' + fileName, crr_path + "/InitDataset/bdd100k/bdd_TL")
                except OSError as error:
                    print(error)
                      
        val_TL = pd.read_csv(crr_path + "/InitDataset/bdd100k/val_df_trafficLight.csv", sep=",", header=0, index_col=0)
        train_TL = pd.read_csv(crr_path + "/InitDataset/bdd100k/train_df_trafficLight.csv", sep=",", header=0, index_col=0)
        
        allAnnotationTL = train_TL.append(val_TL)
        allAnnotationTL.to_csv(crr_path + "/InitDataset/bdd100k/allAnnotationTL.csv")                     

        delete_csvFiles = ["val_df_trafficLight.csv", "train_df_trafficLight.csv"]
        
        for csvFiles in delete_csvFiles:
            os.remove(csv_path + csvFiles)
            
        print("Finished initializing bdd100kTL!")
            
    def parse_label_TL(self, img_annotations, img_width, img_height):
        labels = []
        
        for _, row in img_annotations.iterrows():
            
            img_class = self.config.classes_dict[row["Annotation tag"]]
            light_class = self.config.classes_dict[row["Annotation LightTag"]]
            
            x_left = float(row["x Left"])
            x_right = float(row["x Right"])
            y_lower = float(row["y Bottom"])
            y_upper = float(row["y Upper"])
            
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
            labels.append(
                self.config.formatter.parse_label(
                    light_class,
                    img_width,
                    img_height,
                    x_left,
                    y_lower,
                    x_right,
                    y_upper,
                )
            )

        return labels
    
    # ================================================
    # parses dataset into output folder
    def parse_dataset(self):

        crr_path = os.path.abspath(os.getcwd()) 
        
        annotation_path = crr_path + "/InitDataset/bdd100k/bdd100k/labels/"
        
        image_path = crr_path + "/InitDataset/bdd100k/bdd100k/images/100k/"
        
        # Initialize variables
        x_Right = 0; x_Left = 0; y_Upper = 0; y_Bottom = 0; key= ""; label_in_dictionary = False;

        jpg_name = ""; category = ""; trafficLightColor=""
        
        # Prefix is used to distinguish val / train files
        prefix = ""
        
        # new_row is used to add it to a dataframe
        new_row = "" 
        
        # Creating a folder for TL. This will only contain image files that have traffic lights
        try:
            os.mkdir(crr_path + "/InitDataset/bdd100k/bdd_TL")
        except OSError as error:
            print(error)
            
        # Store paths
        annotation_path = crr_path + "/InitDataset/bdd100k/bdd100k/labels/"        
        image_path_100k = crr_path + "/InitDataset/bdd100k/bdd100k/images/100k/"            
        image_path = crr_path + "/InitDataset/bdd100k/bdd100k/images"
        csv_path = crr_path + "/InitDataset/bdd100k/"
        #*           Start creating annotation file          *#
        
        for infiles in sorted(glob.glob(annotation_path + '*.json')):
            
            # Create dataframes for traffic Light annotation files        
            df_trafficLight = pd.DataFrame(columns =['Filename', 'Annotation tag', 'x Left', 'y Upper', 'x Right',
                              'y Bottom'] )
                        
            if "val" in infiles:
                prefix = "val"
            if "train" in infiles:
                prefix = "train"
            
            with open (infiles, 'r') as myfile:
                for myline in myfile:
                    
                    # Extract file name
                    if "name" in myline:
                        jpg_name = myline.split("\"")[3]
                    
                    # Extract category. Category contains "traffic light"
                    # We are only going to use "traffic light" 
                    if "category" in myline:
                        category = myline.split("\"")[3]
                    
                    # Extract x1, x2, y1, y2 values
                    if category == "traffic light": 
                        if "x1" in myline:
                            x_Left = myline.split("\"")[2]
                            x_Left = x_Left.replace(',', '')
                            x_Left = x_Left.replace(':', '')
                        if "x2" in myline:                    
                            if "box2d" not in myline:
                                x_Right = myline.split("\"")[2]
                                x_Right = x_Right.replace(',', '')
                                x_Right = x_Right.replace(':', '')
                        if "y1" in myline:
                            y_Bottom = myline.split("\"")[2]
                            y_Bottom = y_Bottom.replace(',', '')
                            y_Bottom = y_Bottom.replace(':', '')
                        if "y2" in myline:
                            y_Upper = myline.split("\"")[2] 
                            y_Upper = y_Upper.replace(',', '')
                            y_Upper = y_Upper.replace(':', '')      
                            
                            # After we get y2, new jpg or category starts
                            # So we create a new row and add it to the dataframe
                            
                            # Creating a new row that contains filename, annotation tag, x, y values
                            if category == "traffic light":
                                if (trafficLightColor != 'none'):                                    
                                    new_row = {'Filename': jpg_name, 'Annotation tag': "trafficLight",
                                                'Annotation LightTag': "trafficLight" + trafficLightColor,
                                                'x Left' :x_Left, 'y Upper' :y_Upper, 'x Right':x_Right,
                                                'y Bottom' : y_Bottom}
                                    df_trafficLight = df_trafficLight.append(new_row, ignore_index = True)

                                category = "";
                            y_Upper = 0
                            
                        if category == "traffic light":
                                if "trafficLightColor" in myline:
                                    trafficLightColor = myline.split("\"")[3]
            
            # Create annotation.csv for traffic light
            df_trafficLight.to_csv(crr_path + "/InitDataset/bdd100k/" + prefix + "_df_trafficLight.csv")
            df_TL_no_dup = df_trafficLight.drop_duplicates(subset=['Filename'])

            # Copy necessary files and move them to bdd_TL folder. 
            
            for fileName in df_TL_no_dup['Filename']:
                try:
                    shutil.copy(image_path_100k + prefix + '/' + fileName, crr_path + "/InitDataset/bdd100k/bdd_TL")
                except OSError as error:
                    print(error)
                      
        val_TL = pd.read_csv(crr_path + "/InitDataset/bdd100k/val_df_trafficLight.csv", sep=",", header=0, index_col=0)
        train_TL = pd.read_csv(crr_path + "/InitDataset/bdd100k/train_df_trafficLight.csv", sep=",", header=0, index_col=0)
        
        allAnnotationTL = train_TL.append(val_TL)
        allAnnotationTL.to_csv(crr_path + "/InitDataset/bdd100k/allAnnotationTL.csv")                     

        delete_csvFiles = ["val_df_trafficLight.csv", "train_df_trafficLight.csv"]
        
        for csvFiles in delete_csvFiles:
            os.remove(csv_path + csvFiles)
            
        print("Finished initializing bdd100kTL!")
        
        crr_path = os.path.abspath(os.getcwd()) 
        
        annoteFile_path = crr_path + "/InitDataset/bdd100k/"
        
        annoteFiles = ["allAnnotationTL.csv"]
                
        imagePath_TL = crr_path + "/InitDataset/bdd100k/bdd_TL/"
                
        image_path = [imagePath_TL]
    
        for i in range (len(annoteFiles)):
            
            assert os.path.isfile(annoteFile_path + annoteFiles[i])
            
            print("Started Parsing bdd100kTL!")
            
            annot_df = pd.read_csv(annoteFile_path + annoteFiles[i], sep=",", header=0)
                    
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
            
                assert os.path.isfile(image_path[i] + img_file)
                
                image = read_img(image_path[i] + img_file, self.config.color_mode)
                
                img_width, img_height = get_img_dim(image)
                
                # Grab all labels for this image.
                img_annotations = annot_df[annot_df["Filename"] == img_file]            
                
                labels = self.parse_label_TL(img_annotations, img_width, img_height)
                
                # write out labels and image
                filename = self.prepend + img_file.split("/")[-1]
                
                filename = filename.replace(".jpg", "")
                self.config.formatter.write_data(filename, image, labels, (img_file in img_train))
    
            print("Finished parsing bdd100kTL")
