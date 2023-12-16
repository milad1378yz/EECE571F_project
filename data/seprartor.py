import pandas as pd
import numpy as np
from PIL import Image
import os
# load csv file

data = pd.read_csv("FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv",sep=';')
# get the list of the name of the columns
col_list = data.columns

# get the unique list of "Plane" column
unique_list = data.Plane.unique()
print(unique_list)

included_folders = [
        "Fetal brain_Trans-cerebellum",
        "Fetal brain_Trans-thalamic",
        "Fetal brain_Trans-ventricular",
        "Fetal abdomen",
        "Fetal femur",
        "Fetal thorax",
        "Fetal brain"
    ]

for i in range(len(data)):
    if data.Plane[i] not in included_folders:
        continue
    if data.Plane[i] == "Fetal brain":
        if data.Brain_plane[i] == "Other":
            continue
        # check if the folder exists
        if not os.path.exists("seperated_data/Fetal brain_"+data.Brain_plane[i]+"/"):
            os.makedirs("Fetal brain_"+data.Brain_plane[i]+"/")
        Image.open("FETAL_PLANES_ZENODO/Images/"+data.Image_name[i]+".png").save("seperated_data/Fetal brain_"+data.Brain_plane[i]+"/"+data.Image_name[i]+".png")
    else:
        # check if the folder exists
        if not os.path.exists("seperated_data/"+data.Plane[i]+"/"):
            os.makedirs(data.Plane[i]+"/")
        Image.open("FETAL_PLANES_ZENODO/Images/"+data.Image_name[i]+".png").save("seperated_data/"+data.Plane[i]+"/"+data.Image_name[i]+".png")

    