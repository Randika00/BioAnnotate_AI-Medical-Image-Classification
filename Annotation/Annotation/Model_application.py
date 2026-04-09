import argparse
from h5py.h5a import exists
import testing_genus
import testing_image_number
import testing_skin
import testing_age
import testing_image_type
import testing_image_color
import testing_image_sex
import testing_image_anatomy

import shutil
import os

parser = argparse.ArgumentParser(description='Predicting App')
parser.add_argument('-i')
parser.add_argument('-o')
args = parser.parse_args()

input_image = args.i
output_file = args.o

print(input_image)
print(output_file)

if not os.path.exists(output_file):
    print(output_file)
    shutil.copy("Output.json",output_file)


testing_genus.genus_testing(input_image,output_file)

testing_image_number.image_number_testing(input_image,output_file)

testing_skin.skin_testing(input_image,output_file)

testing_age.age_testing(input_image,output_file)

testing_image_type.image_type_testing(input_image,output_file)

testing_image_color.image_color_testing(input_image,output_file)

testing_image_sex.image_sex_testing(input_image,output_file)

testing_image_anatomy.image_anatomy_testing(input_image,output_file)