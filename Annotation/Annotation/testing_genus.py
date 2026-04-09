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
    model.add(Dense(4, activation='softmax'))
    # The Final layer with two outputs for two categories

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    model.load_weights('genus_classification_model_vgg16.h5')

    return model

def genus_testing(input_image,output_file):

    model = load_cnn()

    category = {0: 'Not Applicable', 1: 'Unknown', 2: 'Person', 3: 'Non-person'}

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

    # print(keys)

    if ("genus_predict" in keys):
        data1["genus_predict"] = str(category[label])

    if ("genus_predict_perc" in keys):
        data1["genus_predict_perc"] = str(prob) + '%'

    with open(output_file, "w") as jsonFile:
        json.dump(data1, jsonFile, indent=2)

