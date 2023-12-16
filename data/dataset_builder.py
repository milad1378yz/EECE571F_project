import pandas as pd
import numpy as np
import random
import argparse
import string

# Create the argument parser
parser = argparse.ArgumentParser()

# Add the boolean argument
parser.add_argument('--do_random_string', action='store_true', help='Whether to do random string')

# Parse the arguments
args = parser.parse_args()

# Access the value of the boolean argument
do_random_string = args.do_random_string
# load csv file

data = pd.read_csv("FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv",sep=';')
# get the list of the name of the columns
col_list = data.columns

# get the unique list of "Plane" column
unique_list = data.Plane.unique()
print(unique_list)

# Function to generate a random string of uppercase alphabets
def generate_random_string(length):
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))

random_list  = [generate_random_string(6) for _ in range(len(unique_list))]
# create a dictionary
dict_mapper = {}
for i in range(len(unique_list)):
    dict_mapper[unique_list[i]] = str(random_list[i])


new_data = pd.DataFrame(columns=["image","text"])
for i in range(len(data)):
    if data.Plane[i] == "Fetal Brain":
        if do_random_string:
            text = "XYZFDSGSGR of the SSDVBNHJYJ ("+dict_mapper[data.Brain_plane[i]]+")"
        else:
            text = "Ultrasound image of the "+data.Brain_plane[i]+" plane of the fetal brain"
    else:
        if do_random_string:
            text = "XYZFDSGSGR of the "+dict_mapper[data.Plane[i]]
        else:
            text = "Ultrasound image of the "+data.Plane[i]+" plane"
    # add the new row to the new_data without append function
    new_data.loc[i] = ["data/FETAL_PLANES_ZENODO/Images/"+data.Image_name[i]+".png",text]

    
print(new_data.head())

import json
with open("FETAL_PLANES_ZENODO/dict_mapper.json", "w") as outfile: 
    json.dump(dict_mapper, outfile)
print(new_data.head())

# save the new_data to csv file
new_data.to_csv("FETAL_PLANES_ZENODO/metadata.csv",index=False)


import datasets

dataset_temp = datasets.load_dataset("FETAL_PLANES_ZENODO/", data_files={"train": "metadata.csv"},features=datasets.Features({'image': datasets.Image(),'text': datasets.Value("string")}),split="train")
# save plot the first image with its text
dataset_temp[0]["image"].save("test.png")
print(dataset_temp[0]["text"])
    