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

# Traffic signs for U.S
# Check the signs from the bottom of the page
# https://www.mapillary.com/developer/api-documentation/

# This gitHub contains traffic sign information
# https://github.com/mapillary/mapillary_sprite_source/blob/master/mapping.json

trafficSign = {"regulatory--stop--g1" : "stop",
               "warning--stop-ahead--g1": "stopAhead",
               "regulatory--yield--g1": "yield",
               "warning--yield-ahead--g1" : "yieldAhead",
               "information--pedestrians-crossing--g2" : "pedestrianCrossing",
               "warning--traffic-signals--g3" : "signalAhead",               
                "regulatory--maximum-speed-limit-10--g3": "speedLimit",
                "regulatory--maximum-speed-limit-15--g3": "speedLimit",
                "regulatory--maximum-speed-limit-20--g3": "speedLimit",
                "regulatory--maximum-speed-limit-25--g3": "speedLimit",
                "regulatory--maximum-speed-limit-30--g3": "speedLimit",
                "regulatory--maximum-speed-limit-35--g3": "speedLimit",
                "regulatory--maximum-speed-limit-40--g3": "speedLimit",
                "regulatory--maximum-speed-limit-45--g3": "speedLimit",
                "regulatory--maximum-speed-limit-50--g3": "speedLimit",
                "regulatory--maximum-speed-limit-55--g3": "speedLimit",
                "regulatory--maximum-speed-limit-60--g3": "speedLimit",
                "regulatory--maximum-speed-limit-70--g3": "speedLimit",
                "regulatory--maximum-speed-limit-75--g3": "speedLimit",
                "regulatory--maximum-speed-limit-80--g3": "speedLimit",
                "regulatory--maximum-speed-limit-85--g3": "speedLimit",
                "complementary--maximum-speed-limit-10--g1": "speedLimit",
                "complementary--maximum-speed-limit-15--g1": "speedLimit",
                "complementary--maximum-speed-limit-20--g1": "speedLimit",
                "complementary--maximum-speed-limit-25--g1": "speedLimit",
                "complementary--maximum-speed-limit-30--g1": "speedLimit",
                "complementary--maximum-speed-limit-35--g1": "speedLimit",
                "complementary--maximum-speed-limit-40--g1": "speedLimit",
                "complementary--maximum-speed-limit-45--g1": "speedLimit",
                "complementary--maximum-speed-limit-50--g1": "speedLimit",
                "complementary--maximum-speed-limit-55--g1": "speedLimit",
                "complementary--maximum-speed-limit-60--g1": "speedLimit",
                "complementary--maximum-speed-limit-65--g1": "speedLimit",
                "complementary--maximum-speed-limit-70--g1": "speedLimit",
                "complementary--maximum-speed-limit-75--g1": "speedLimit",
                "complementary--maximum-speed-limit-80--g1": "speedLimit",
                "complementary--maximum-speed-limit-85--g1": "speedLimit",
                "complementary--maximum-speed-limit-90--g1": "speedLimit",
                "complementary--maximum-speed-limit-95--g1": "speedLimit",
                "complementary--turn-right--g1": "turnRight",
                "complementary--turn-right--g2" : "turnRight",
                "complementary--turn-left--g1" : "turnLeft",
                "complementary--turn-left--g2" : "turnLeft",
                "regulatory--turn-right--g3" : "turnRight",
                "regulatory--turn-left--g2" : "turnRight",
                "regulatory--no-right-turn--g1": "noRightTurn",
                "regulatory--no-right-turn--g3": "noRightTurn",
                "regulatory--no-left-turn--g1" : "noLeftTurn",
                "regulatory--no-left-turn--g4" : "noLeftTurn",
                "regulatory--do-not-pass--g1" : "doNotPass",
                "warning--no-passing-zone--g1" : "doNotPass",
                "warning--no-passing-zone--g2" : "doNotPass",
                "regulatory--roundabout--g2" : "roundabout",
                "regulatory--roundabout--g3" : "roundabout",
                "warning--roundabout--g2" : "roundabout",
                "regulatory--circular-intersection--g1" : "intersection",
                "regulatory--circular-intersection--g2" : "intersection",
                "regulatory--circular-intersection--g3" : "intersection",
                "regulatory--circular-intersection--g4" : "intersection",
                "warning--curve-out-intersection-left--g1" : "intersection",
                "warning--curve-out-intersection-right--g1" : "intersection",
                "warning--railroad-intersection--g1" : "intersection",
                "warning--railroad-intersection--g2" : "intersection",
                "warning--railroad-intersection--g3" : "intersection",
                "warning--railroad-intersection--g4" : "intersection",
                "warning--railroad-intersection--g5" : "intersection",
                "regulatory--keep-right--g3": "keepRight",
                "regulatory--keep-right--g4": "keepRight",
                "regulatory--keep-right--g5": "keepRight",
                "regulatory--keep-right--g6": "keepRight",
                "regulatory--keep-right--g7": "keepRight",
                "regulatory--keep-right--g8": "keepRight",
                "regulatory--keep-right--g9": "keepRight",
                "warning--keep-right--g1": "keepRight",
                "warning--keep-left--g1": "keepLeft",
                "regulatory--keep-left--g2" : "keepLeft",
                "regulatory--keep-left--g3" : "keepLeft",
                "regulatory--keep-left--g4" : "keepLeft",
                "regulatory--keep-left--g5" : "keepLeft",
                "regulatory--keep-left--g6" : "keepLeft",
                "regulatory--keep-left--g7" : "keepLeft",
                "warning--curve-left--g2" : "curveLeft",
                "warning--curve-left--g3" : "curveLeft",
                "warning--curve-right--g2" : "curveRight",
                "warning--curve-right--g3" : "curveRight",
                "warning--slow--g1" : "slow",
                "warning--school-zone--g2" : "school",
                "warning--traffic-merges-left--g2": "merge",
                "warning--traffic-merges-left--g3": "merge",
                "warning--traffic-merges-left--g4": "merge",
                "warning--traffic-merges-right--g1": "merge",
                "warning--traffic-merges-right--g2": "merge",
                "warning--traffic-merges-right--g3": "merge",
                "warning--entering-roadway-merge--g1": "merge",
                "warning--entering-roadway-merge--g2" : "merge"}

class MTSD(Dataset):
    def __init__(self, mtsd_config: dict, config: Common):
        """
        mtsd_config: dict
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
        self.raw_path_train_0 = config.root + mtsd_config["RAW_FOLDER"] + mtsd_config["TRAIN_ZIPS"][0]
        self.raw_path_train_1 = config.root + mtsd_config["RAW_FOLDER"] + mtsd_config["TRAIN_ZIPS"][1]
        self.raw_path_train_2 = config.root + mtsd_config["RAW_FOLDER"] + mtsd_config["TRAIN_ZIPS"][2]
        self.raw_path_Annotation = config.root + mtsd_config["RAW_FOLDER"] + mtsd_config["ANNOTATION_ZIP"]  
        self.raw_path_val = config.root + mtsd_config["RAW_FOLDER"] + mtsd_config["VAL_ZIP"]
        self.raw_path_test = config.root + mtsd_config["RAW_FOLDER"] + mtsd_config["TEST_ZIP"]
        
        # path to allAnnotation file
        self.annot_file = os.path.abspath(os.getcwd())  + "/InitDataset/MTSD/allAnnotations.csv"
        
        # path to initialization folder (absolute)
        self.init_path = config.init_folder + mtsd_config["INIT_FOLDER"]
                
        self.prepend = mtsd_config["PREPEND"]
        
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

        print("Started initializing MSTD Dataset!")
        
        #`      Unzip files         `#
        
        # List of the zipped files. 
        zip_list = [self.raw_path_train_0, self.raw_path_train_1, self.raw_path_train_2,
                    self.raw_path_val, self.raw_path_Annotation]

        for i in range (len(zip_list)):
            MTSD.unzip_func(self, zip_list[i])
         
        #       Unzip files End     `#    
            
        # Create a dataframe to store all annotations. Format will be similar to LISATL and LISTTS
        df = pd.DataFrame(columns =['Filename', 'Annotation tag', 'X max', 'Y max', 'X min', 'Y min'] )


        #'      Creating paths      '#        
        
        # Current path ~/traffic_detection_dataset
        crr_path = os.path.abspath(os.getcwd()) 
        
        init_path = crr_path + "/InitDataset/MTSD"

        # Path for annotation dir
        path = init_path + "/mtsd_v2_fully_annotated/annotations/"

        #'      Creating paths End  '#
        
        
        #`      Initialize Variables    `#
        # Label in dictionary will become true if a file contains any traffic signs that are in trafficSign dictionary (Please check line 15)
        xmin = 0; ymin = 0; xmax = 0; ymax = 0; key= ""; label_in_dictionary = False; sign_data = "";
        
        
        # Create another directory that stores US images
        try:
            os.mkdir(init_path + "/US_images")            
        except OSError as error: #If images are not in the directory
            error
                        
        # json file contains information for images 
        # We are going to extract fileName, x and y values, label
        
        #`      Generating allAnnotation.csv file     `#
        
        # 2 Json files will be parsed(Train and Val)
        
        for infiles in sorted(glob.glob(path + '*.json')):
            with open (infiles, 'r') as myfile:
                for myline in myfile:
                    
                    # Each file contains label, we are going to extract images by labels 
                    if "label" in myline:
                        sign_data = myline.split("\"")[3]
                        res = dict(filter(lambda item: sign_data in item[0], trafficSign.items()))
                        
                        # If a json file contains target label (ex. speet limit) then we are going to move the image to US_images folder
                        if (len(res) != 0):
                              label_in_dictionary = True;            
                              
                    # If a file contains target label, add the file to dataframe
                    if(label_in_dictionary):
                        if "xmin" in myline:
                            get_xmin = myline.split(" ")
                            extract_xmin = get_xmin[-1]
                            xmin = extract_xmin.replace(',', '')
                            
                        if "ymin" in myline:
                            get_ymin = myline.split(" ")
                            extract_ymin = get_ymin[-1]
                            ymin = extract_ymin.replace(',', '')

                        if "xmax" in myline:
                            get_xmax = myline.split(" ")
                            extract_xmax = get_xmax[-1]
                            xmax = extract_xmax.replace(',', '')
                            
                        if "ymax" in myline:
                            get_ymax = myline.split(" ")
                            extract_ymax = get_ymax[-1]
                            ymax = extract_ymax.replace(',', '')
                            Extract_fileName = infiles.split("/")
                            fileName = Extract_fileName[-1].replace("json", "jpg") 
                            
                            # Check if the file actually exists
                            if(os.path.exists(init_path + "/images/" + fileName)):
                                new_row = {'Filename': "US_images/" + fileName, 'Annotation tag': res[sign_data], 
                                            'X max' :xmax, 'Y max' :ymax, 'X min':xmin,
                                            'Y min' : ymin}
                                df = df.append(new_row, ignore_index = True)
                                
                                # Move U.S images from the image folder to US_images folder
                                try:
                                    shutil.move(init_path + "/images/" + fileName, init_path + "/US_images/"+ fileName)
                                except OSError as error:
                                    print(error)

                            label_in_dictionary = False
        
                
        # Create allAnnotations.csv file                                
        df.to_csv(init_path + "/allAnnotations.csv")
        
        
        #`      Generating allAnnotation.csv file Ends      `#
        
        # Remove unused image files.
        shutil.rmtree(init_path + "/images")
        
        print("Finished initializing MSTD Dataset!")
    
    def parse_label(self, img_annotations, img_width, img_height):
        labels = []

        for _, row in img_annotations.iterrows():
            img_class = self.config.classes_dict[row["Annotation tag"]]
            x_left = float(row["X min"])
            x_right = float(row["X max"])
            y_lower = float(row["Y min"])
            y_upper = float(row["Y max"])
            
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
        print("Started Parsing MTSD")
        annot_df = pd.read_csv(self.annot_file, sep=",", header=0)
                
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
            
            filename = filename.replace(".jpg", "")
            
            self.config.formatter.write_data(filename, image, labels, (img_file in img_train))


        print("Finished parsing MTSD")