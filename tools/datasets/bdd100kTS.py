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

# Reference for the Yolo v4 classifier function
# https://github.com/quangnhat185/Machine_learning_projects/blob/master/YOLOv4_traffic_signs_detection/YOLOV4_traffic_sign_detection.ipynb
# https://cloudxlab.com/blog/object-detection-yolo-and-python-pydarknet/


class bdd100kTS(Dataset):
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
    
    def Yolov4_classifier(INPUT_FILE):
        crr_path = os.path.abspath(os.getcwd())
        OUTPUT_FILE= 'predicted.jpg'
        LABELS_FILE= crr_path + "/Run2/classes.names"
        CONFIG_FILE= crr_path + "/Run2/v4tiny-512.cfg"
        WEIGHTS_FILE= crr_path + "/Run2/v4tiny-512_best.weights"
        CONFIDENCE_THRESHOLD=0.95
        
        def image_preprocess(image):
            
            text_arr = []
            
            LABELS = open(LABELS_FILE).read().strip().split("\n")
        
            np.random.seed(4)
            COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                dtype="uint8")
        
        
            net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
        
            image = cv2.imread(INPUT_FILE)
            (H, W) = image.shape[:2]
        
            # determine only the *output* layer names that we need from YOLO
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
            # construct a blob from the input image and then perform a forward pass of the YOLO object detector, 
            # giving us our bounding boxes and associated probabilities
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()
        
            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
        
            # loop over each of the layer outputs
            for output in layerOutputs:
            # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > CONFIDENCE_THRESHOLD:
                # scale the bounding box coordinates back relative to the
        
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
        
                        # use the center (x, y)-coordinates to derive the top and
        
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
        
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
        
            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,CONFIDENCE_THRESHOLD)
        
            # ensure at least one detection exists
            if len(idxs) > 0:
            # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
        
                    color = [int(c) for c in COLORS[classIDs[i]]]
        
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                    text_arr.append(text)

        
            return text_arr
        
        class_name = image_preprocess(INPUT_FILE)
        
        return class_name
            
    def init_dataset(self):
        """
        unzips lisatl into intermidiary INIT_FOLDER
        cleans up files from unzip to minimize disk space.
        """
        
        assert os.path.isfile(self.raw_path)  # makes sure that the zip actually exists.
        print("Started initializing bdd100kTS!")

        # List of the zipped files. 
        # raw_path => image files, annot_file => labels (annotation file)
        
        #*               Unzip files                       *#
        
        zip_list = [self.raw_path, self.annot_file]
                        
        for i in range (len(zip_list)):
            bdd100kTS.unzip_func(self, zip_list[i])
        
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
        
        # Creating a folder for TS and TL. This will only contain image files that have traffic sign or traffic lights
        try:
            os.mkdir(crr_path + "/InitDataset/bdd100k/bdd_TS")
        except OSError as error:
            print(error)
            
        # Store paths
        annotation_path = crr_path + "/InitDataset/bdd100k/bdd100k/labels/"        
        image_path_100k = crr_path + "/InitDataset/bdd100k/bdd100k/images/100k/"            
        image_path = crr_path + "/InitDataset/bdd100k/bdd100k/images"
        csv_path = crr_path + "/InitDataset/bdd100k/"
        #*           Start creating annotation file          *#
        
        for infiles in sorted(glob.glob(annotation_path + '*.json')):
            
            # Create dataframes for traffic Sign annotation files        
            
            df_trafficSign =  pd.DataFrame(columns =['Filename', 'Annotation tag', 'x Left', 'y Upper', 'x Right',
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
                    
                    # Extract category. Category contains "traffic light", "traffic sign", and others.
                    # We are only going to use "traffic light" and "traffic sign"
                    if "category" in myline:
                        category = myline.split("\"")[3]
                    
                    # Extract x1, x2, y1, y2 values
                    if category == "traffic sign": 
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
                            ##### Part for Traffic sign ####       
                            if category == "traffic sign":
                                
                                # If you want to label to trafic sign please use this part
                                #'              Label to traffic sign              `#
                                
                                # new_row = {'Filename': jpg_name, 'Annotation tag': category, 
                                #             'x Left' :x_Left, 'y Upper' :y_Upper, 'x Right':x_Right,
                                #             'y Bottom' : y_Bottom}
                                # df_trafficSign = df_trafficSign.append(new_row, ignore_index = True)
                                
                                #'              Label to traffic sign Ends         `#
                                
                                # If you want to label using Yolo v4 Tiny classifier, please use this part
                                
                                #`              Label using Yolo v4.tiny model     `#
                                
                                class_name = bdd100kTS.Yolov4_classifier(image_path_100k + prefix + "/" + jpg_name)
                                
                                for x in class_name:
                                    new_row = {'Filename': jpg_name, 'Annotation tag': x.split(':')[0], 
                                                'x Left' :x_Left, 'y Upper' :y_Upper, 'x Right':x_Right,
                                                'y Bottom' : y_Bottom}
                                    df_trafficSign = df_trafficSign.append(new_row, ignore_index = True)
                                    
                                #`              Label using Yolo v4.tiny model ENDS `#
                                    
                                
                                category = "";
                            y_Upper = 0
                            
            # Create annotation.csv for traffic light and traffic sign
            
            # Copy necessary files and move them to bdd_TS folder. 
  
            df_trafficSign.drop_duplicates(subset=['Filename', 'Annotation tag'], keep='first', inplace=True)
            df_trafficSign.to_csv(crr_path + "/InitDataset/bdd100k/"  + prefix + "_df_trafficSign.csv")
            df_TS_no_dup = df_trafficSign.drop_duplicates(subset=['Filename'])

            for fileName in df_TS_no_dup['Filename']:
                try:
                    shutil.copy(image_path_100k + prefix + '/' + fileName, crr_path + "/InitDataset/bdd100k/bdd_TS")
                except OSError as error:
                    print(error)
        
        val_TS = pd.read_csv(crr_path + "/InitDataset/bdd100k/val_df_trafficSign.csv", sep=",", header=0, index_col=0)
        train_TS = pd.read_csv(crr_path + "/InitDataset/bdd100k/train_df_trafficSign.csv", sep=",", header=0, index_col=0)
        
        allAnnotationTS = train_TS.append(val_TS)
        allAnnotationTS.to_csv(crr_path + "/InitDataset/bdd100k/allAnnotationTS.csv", index=False)        
        
        #*           Creating annotation file Ends         *#
        
        # delete some of the unnecessary images to free up spaces.
        
        #shutil.rmtree(image_path)
         
        delete_csvFiles = ["val_df_trafficSign.csv", "train_df_trafficSign.csv"]
        
        for csvFiles in delete_csvFiles:
            os.remove(csv_path + csvFiles)
            
        print("Finished initializing bdd100kTS!")
    
    def parse_label_TS(self, img_annotations, img_width, img_height):
        labels = []

        for _, row in img_annotations.iterrows():
            img_class = self.config.classes_dict[row["Annotation tag"]]
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

        return labels
    # ================================================
    # parses dataset into output folder
    def parse_dataset(self):
        
        crr_path = os.path.abspath(os.getcwd()) 
        
        annoteFile_path = crr_path + "/InitDataset/bdd100k/"
                
        annoteFiles = ["allAnnotationTS.csv"]
        
        ##### Image Path for TS #####
        imagePath_TS = crr_path + "/InitDataset/bdd100k/bdd_TS/"
        
        image_path = [imagePath_TS]
        
        for i in range (len(annoteFiles)):
            
            assert os.path.isfile(annoteFile_path + annoteFiles[i])
            
            print("Started Parsing bdd100kTS!")
            
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
                
                # convert class/label to numerical for darknet

                labels = self.parse_label_TS(img_annotations, img_width, img_height)
                
                # write out labels and image
                filename = self.prepend + img_file.split("/")[-1]
                
                filename = filename.replace(".jpg", "")
                self.config.formatter.write_data(filename, image, labels, (img_file in img_train))
    
            print("Finished parsing bdd100kTS")