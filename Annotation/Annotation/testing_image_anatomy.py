from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import cv2, os
import numpy as np
import argparse
import json


def load_cnn():
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(50, 50, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # The first CNN layer followed by Relu and MaxPooling layers

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # The second convolution layer followed by Relu and MaxPooling layers

    model.add(Flatten())
    model.add(Dropout(0.5))
    # Flatten layer to stack the output convolutions from second convolution layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # Dense layer of 64 neurons
    model.add(Dense(58, activation='softmax'))
    # The Final layer with two outputs for two categories

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    model.load_weights('Anatomy_CNN.h5')

    return model

def image_anatomy_testing(input_image,output_file):

    model = load_cnn()

    category = {0:'Abdomen',1:'Ankle',2:'Anus',3:'Arm',4:'Armpit',5:'Back',6:'Breasts',7:'Buccal mucosa',8:'Buttocks',9:'Cheek',10:'Chest',11:'Chin',12:'Ear',13:'Elbow',14:'Eyelids',15:'Eyes',16:'Face',17:'Female genitalia',18:'Finger',19:'Fingernails',20:'Foot',21:'Forehead',22:'Full body',23:'Genitalia',24:'Gingiva',25:'Glabella',26:'Groin',27:'Hair',28:'Hand',29:'Heel',30:'Hip',31:'Knee',32:'Leg',33:'Lips',34:'Mouth',35:'Nail',36:'Nape',37:'Neck',38:'Nose',39:'Palate',40:'Palm',41:'Penis',42:'Scalp',43:'Scrotum',44:'Shin',45:'Shoulder',46:'Sole',47:'Teeth',48:'Temple',49:'Thigh',50:'Toe',51:'Toenails',52:'Tongue',53:'Torso',54:'Wrist',55:'Head',56:'Not applicable',57:'Unknown'}


    #for img_name in img_names:
    img_path = os.path.join(input_image)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (50, 50))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 50, 50, 1))

    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    prob = np.max(result, axis=1)[0]
    prob = round(prob * 100, 2)

    img[:50, :] = [0, 255, 0]

    cv2.putText(img, str(category[label]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    cv2.putText(img, str(prob) + '%', (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)


    if (os.path.exists(output_file)):
        f = open(output_file)
        data1 = json.load(f)


        keys = data1.keys()



    if ("image_anatomy_predict" in keys):
        data1["image_anatomy_predict"] = str(category[label])

    if ("image_anatomy_predict_perc" in keys):
        data1["image_anatomy_predict_perc"] = str(prob) + '%'

    with open(output_file, "w") as jsonFile:
        json.dump(data1, jsonFile, indent=2)

