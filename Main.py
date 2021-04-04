# -*- coding: utf-8 -*-


import os


from generator import segmentation_generator
from get_model import get_model

import numpy as np
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, random_shift,random_shear
import PIL
from PIL import ImageOps
import tensorflow as tf

img_size = (160, 160)
num_classes = 3
batch_size = 32
epochs=15



images_dir="Dataset/images"
target_dir="Dataset/trimaps"


image_paths=sorted(
    os.path.join(images_dir,fname)
    for fname in os.listdir(images_dir)
    if fname.endswith(".jpg")
    
    )


masks_paths=sorted(
    os.path.join(target_dir,fname)
    for fname in os.listdir(target_dir)
    if fname.endswith(".png") and not fname.startswith(".")
    
    )


x_train, x_test, y_train, y_test=train_test_split(image_paths,masks_paths,test_size=0.10, random_state=42)

train_gen=segmentation_generator(batch_size, img_size, x_train, y_train, random_augmentation=False)
test_gen=segmentation_generator(batch_size, img_size, x_test, y_test, random_augmentation=False)



model=get_model(img_size, num_classes)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")


model.fit(train_gen,epochs=epochs, validation_data=test_gen)



val_preds = model.predict(test_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(array_to_img(mask))
    display(img)


# Display results for validation image #10
i = 10

# Display input image
display(Image(filename=x_test[i]))

# Display ground-truth target mask
img = PIL.ImageOps.autocontrast(load_img(y_test[i]))
display(img)

# Display mask predicted by our model
display_mask(i)  # Note that the