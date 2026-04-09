import cv2
import numpy as np
import argparse
import json
import os

def image_color_testing(input_image,output_file):

    #test image
    img=cv2.imread(input_image)

    # splitting b,g,r channels
    b,g,r=cv2.split(img)

    # getting differences between (b,g), (r,g), (b,r) channel pixels
    r_g=np.count_nonzero(abs(r-g))
    r_b=np.count_nonzero(abs(r-b))
    g_b=np.count_nonzero(abs(g-b))

    # sum of differences
    diff_sum=float(r_g+r_b+g_b)

    # finding ratio of diff_sum with respect to size of image
    ratio=diff_sum/img.size

    color_dir = {}


    if (os.path.exists(output_file)):
        f = open(output_file)
        data1 = json.load(f)


        keys = data1.keys()


    if ("image_color_predict" in keys):
        if ratio > 0.005:
            data1["image_color_predict"] = "Colour Image"
            # print("Color Image")
        else:
            data1["image_color_predict"] = "Mono Image"
            # print("Mono Image")


    with open(output_file, "w") as jsonFile:
        json.dump(data1, jsonFile, indent=2)

    print("Successfully created")


