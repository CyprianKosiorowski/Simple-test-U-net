# -*- coding: utf-8 -*-


from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import scipy
from scipy import ndimage 
import random

class segmentation_generator(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, random_augmentation=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.random_augmentation=random_augmentation
    def __len__(self):
        return len(self.target_img_paths) // self.batch_size
    
    def __random_augmentation(image,target):
        angles=[-20,-10,-5,5,10,20]
        angle=random.choice(angles)
        image=ndimage.rotate(image,angle,reshape=False)
        target=ndimage.rotate(target,angle,reshape=False)
        return image, target
    def __getitem__(self,idx):
        i=idx*self.batch_size
        bach_input_img_paths=self.input_img_paths[i:i+self.batch_size]
        batch_input_target_img_paths=self.target_img_paths[i:i+self.batch_size]
        x=np.zeros((self.batch_size,)+self.img_size+(3,),dtype="float32")
        y=np.zeros((self.batch_size,)+self.img_size+(1,),dtype="uint8")
        for j, (path_img, path_tar) in enumerate(zip(bach_input_img_paths,batch_input_target_img_paths)):
            img=img_to_array(load_img(path_img,target_size=self.img_size))
            target=img_to_array(load_img(path_tar,target_size=self.img_size,color_mode="grayscale"))
            if self.random_augmentation==True:
                img,target=self.__random_augmentation(img,target)
            x[j]=img
           # y[j]=np.expand_dims(target,2)
            y[j]=target
            y[j]-=1
        return x, y