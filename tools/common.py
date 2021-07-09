import os
import shutil

from .formats.darknet import Darknet


def yes_no():
    yes = set(['yes','y', 'ye', ''])
    no = set(['no','n'])
    while True:
        choice = input("[y/n]: ").lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
            exit(0)
        else:
            print("Please respond with [y/n]")



class Common:
    def __init__(self, config_options):
        if config_options["ROOT"] == "":
            self.root = os.getcwd() + "/"
        else:
            self.root = config_options["ROOT"]
            
        # creates and loads absolute path to names/classes
        self.names_file = self.root + config_options["NAMES"]
        with open(self.names_file, "r") as file:
            self.classes = file.read().split("\n")
            self.classes_dict = {self.classes[x]: x for x in range(len(self.classes))}

        # other file options.
        self.init_folder: str = self.root + config_options["INIT_DATA"]
        self.output_folder: str = self.root + config_options["OUTPUT_DATA"]
        self.format: str = config_options["FORMAT"]

        self.test_split: float = config_options["TEST_SPLIT"]
        self.random_state: int = config_options["RANDOM_STATE"]
        self.show_img: bool = config_options["SHOW_IMG"]
        self.color_mode: int = config_options["COLOR_MODE"]
        self.image_format: str = config_options["IMG_FORMAT"]

        # handles format chooser
        if(self.format == "DARKNET"):
            self.formatter = Darknet(self.output_folder, self.image_format)
        elif(self.format == "COCO"):
            print("Output format not yet supported!")
            exit(0)
        elif(self.format == "VOC"):
            print("Output format not yet supported!")
            exit(0)
        else:
            print("Output format not recognized!")
            exit(0)



    def create_folders(self, init=True):
        folder = self.init_folder if init else self.output_folder

        if not os.path.isdir(folder):
            # folder doesn't exist. Make folder
            os.makedirs(folder, exist_ok=True)
            return
        else:
            if not os.listdir(folder):
                # folder exists, but is empty.
                pass
            else:
                # folder exists and is not empty. Prompt user.
                print("Folder:", folder, "exists and is not empty.")
                print("Continuing the process will delete all existing files. Proceed?")
                ans = yes_no()

                if not ans:
                    print("exiting...")
                    exit(0)
                    # user responded no. end program
                else:
                    # repsonded yes. Delete folder and recreate.
                    shutil.rmtree(folder)
                    os.makedirs(folder, exist_ok=True)

        if not init:
            self.formatter.init_output()