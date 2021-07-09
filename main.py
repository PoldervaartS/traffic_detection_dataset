import yaml
import logging
import sys
import argparse

from tools.common import Common
from tools.datasets.lisats import LISA_TS
from tools.datasets.lisatl import LISA_TL
from tools.datasets.mtsd import MTSD
from tools.datasets.bdd100kTL import bdd100kTL
from tools.datasets.bdd100kTS import bdd100kTS

def main():

    parser = argparse.ArgumentParser(description= "Used to process large files.")
    parser.add_argument("-i", "--initialize", action="store_true", help="initialize datasets into intermediate before final processing.")


    args = parser.parse_args()


    if args.initialize:
        print("Preparing to initialize datasets...")
    else:
        print('Preparing to parse datasets...')


    # loads config file
    with open(r"dataset.yaml") as file:
        input_dict = yaml.load(file,  Loader=yaml.FullLoader)

   
    config = Common(input_dict["DATA_CONFIG"])
    config.create_folders(args.initialize)

    datasets = [
        LISA_TS(input_dict["LISA_TS"], config)
        # LISA_TL(input_dict["LISA_TL"], config)      
        # bdd100kTL(input_dict["bdd100k"], config)
        # bdd100kTS(input_dict["bdd100k"], config)
        # LISA_TL(input_dict["LISA_TL"], config)
        # MTSD(input_dict["MTSD"], config)
        # TODO: other datasets are also initialized here.
    ]


    for dataset in datasets:
        if args.initialize:
            dataset.init_dataset()
        else:
            dataset.parse_dataset()


if __name__ == "__main__":
    main()
    