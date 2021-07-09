# About 

GM Autodrive Challenge is a competition inviting various research focused schools in competition to develop the best autonomous vehicle. This repository represents the dataset aggregation for Traffic Sign and Signal recognition 2021 utilizing already made datasets such as LISA and BDD100k, reformatting them to have an identical common format, and then utilizing that to train the models referenced in the traffic_detect repo. Worked alongside Angelo Yang https://www.linkedin.com/in/ryang42, Nicholas Vacek https://www.linkedin.com/in/jung-hoon-seo-5ab813178, and Jung Hoon Seo https://www.linkedin.com/in/nicholas-vacek-314a23187
We ultimately managed to place 3rd in great success and more information can be found here! http://autodrive.tamu.edu/ 

# Final Demo
- [Final Demo](https://drive.google.com/file/d/1ASL5h8EOw2rj-DX_UaIL8lclbXFhTwpL/view?usp=sharing)
    - You will need to access the link with TAMU email 
# The GIGA Traffic signs and signals dataset (US)
- [LISA Traffic Sign Dataset (~8gb zip)](http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html)
    - Download link is at the bottom of the page.
- [LISA Traffic Lights Dataset (~5gb zip)](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset)
    - The LISA website is having trouble and is down for the Traffic Lights dataset, so linked above is kaggle mirror.
- [Mapillary Traffic Sign Dataset (~42gb over 6 zips)](https://www.mapillary.com/dataset/trafficsign)
    - Download links are sent to your email and you have 5 days to download everything. Download only the fully_annotated zips (`test`, `train0`-`2`, `val`, `annotation`)
- [BDD100k Traffic Sign and Light Dataset (~6.94 GB)](https://bdd-data.berkeley.edu/)
    - You need to sign up for this website to download the dataset
    - After you signing up, click Download Dataset -> login -> Click 'Download' from the left panel -> Find BDD100K section and click 'Images' and 'Labels' buttons to download two zip files

## Requirements
- Python 3
    - opencv (cv2)
    - numpy
    - pandas
    - yaml
    - matplotlib
    - sklearn (for train_test_split)
    - jupyter notebook (if you want see data explorations)
- LOTS of space!
- Fast storage is recommended

## Usage
- Make sure to clone this in some place where you have **LOTS** of storage, like **>100gb!**
    - The size requirement is temporary! Once initialized it should only be ~20gb or so.
- Place all downloaded zips into the `RAW_DATASETS/` folder
    - There is no need to rename any of the zips, or put in subfolders! If you do make any of those changes, you will need to change the location in `dataset.yaml`
- Edit the parameters in `dataset.yaml` to your liking.
    - make sure you don't forget the trailing `/` for any paths to folders!
- run `python main.py --initialize`
    - Once it is finished you can delete the `.zip`s in the `RAW_DATASETS` folder if you need space back, otherwise just keep it in case you need it again.

### Formats:
- [Darknet format](https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885)

## Notes:
- Currently outputs in Darknet format for use in training with YOLO. This can easily be converted to COCO or Pascal VOC format for other classifiers.
    - TODO: add option for other formats.
---

## LISA Traffic Sign Dataset Notes
Authors claim *47* different classes, but most of them are extraneous. A simple cleaning usually brings it down to just 35. Further removing some of the classes we don't need ends for the autodrive competition nets us just 26 classes.
Notes:
- the `allAnnotations.csv` supplied by the author is deliminated by both commas and semicolons. If you plan to manually look through this dataset, you need to convert all semicolons to commans
    - You can open csv in something like `notepad++` and just find and replace, or
    - Linux: run `sed -i 's/;/,/g' <path_to>/allAnnotations.csv`
- The authors include a `negatives/` folder with images that don't have any signs.
    - This isn't necessary if for training YOLO since they can figure out what is negative with the empty space in most images.
    - For other detectors, please do research and see if it is recommended or not.

## LISA Traffic Light Dataset Notes

LISA traffic light dataset contains around 44000 images, and there are 6 directories, nightTrain, nightSequence1,2, dayTrain, and daySequence1,2, with two annotation files, frameAnnotationsBULB and frameAnnotationsBOX, for each directory. We are only using frameAnnotationsBOX for bounding boxes.

Notes:
- When you run 'python main.py -- initialize', it will remove unused directories and generate allAnnotation.csv file. 
- After the initialization, run 'python main.py' to generate annotation txt files for each image.

## MTSD Traffic Sign Dataset Notes
Please follow the instruction above, The GIGA Traffic signs and signals dataset (US), to download MTSD dataset. The dataset contains 20% North America and 80% of other countries, and we tried to extract the traffic signs that are related to U.S. You can check the category of the traffic signs from https://www.mapillary.com/developer/api-documentation/ and the country distributions for the traffic sign https://github.com/mapillary/mapillary_sprite_source/blob/master/mapping.json.

Usage
- Move the following zip files to the RAW_DATASETS folder
  - mtsd_v2_fully_annotated_images.train.0.zip
  - mtsd_v2_fully_annotated_images.train.1.zip
  - mtsd_v2_fully_annotated_images.train.2.zip
  - mtsd_v2_fully_annotated_images.val.zip
  - mtsd_v2_fully_annotated_annotation.zip
- Run python main.py --initialize
- Run python main.py

## BDD100k Traffic Sign and Traffic Light Dataset Notes
Please follow the instruction above, The GIGA Traffic signs and signals dataset (US), to download the BDD100K dataset
- After downloading two zip files, bdd100k_labels_release.zip (6.94 GB) and bdd100k_images.zip (115.5 MB), put these two zip files to RAW_DATASETS
- Run python main.py --initialize
- It will take more than 2 hours to generate annotation files because there are more than 70,000 images
- Run main.py after finishing the initialization

Notes:
BDD100K Traffic Light:
  - BDD100k has traffic light colors' information, Green, yellow, red, and none. Only three labels, TrafficLightgreen, TrafficLightyellow, and TrafficLightred, are used.

Notes for the traffic sign:
  - Please download the weights from [Yolo Weights](https://drive.google.com/drive/folders/1JYdu6nVTAw-i4xpqJ-Zrzcw2Wiwc3alU)
  - Current weights are from YOLOv4 Weights -> YOLOV4-TINY -> Run2
  - Due to the limitation of YOLOv4. Tiny model, we recommend using YOLOv4-CSP (But it will be a lot slower without a GPU)
  - Please move the weight folder to traffic_detection_dataset folder
  - You might need to update line 67 - 70 only if you use CSP model
