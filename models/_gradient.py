from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from ._classifier import Classifier


class GradientClassifier(Classifier):
    def __init__(self, stride):
        self.stride = stride
        self.clf =  KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        self.df_train = None
        self.df_test = None
    
    def calculate_features(self, image_folder_path, window_half_height=4, image_height=80):
        window_height = window_half_height * 2
        
        folder_path = Path(image_folder_path)
        df = pd.DataFrame(columns=['photo_id', 'face_id', *list(range(1, int(np.ceil((image_height - window_height) / self.stride + 1))))])
        for image_path in folder_path.glob('*.jpg'):
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            
            features = self.__get_gradient(image, window_half_height, self.stride, image_height)
            
            photo_id, face_id = map(int, image_path.stem.split('_'))
            row = [photo_id, face_id, *features]
            df = pd.concat([df, pd.DataFrame([row], columns=df.columns)])

        df = df.astype('float64')
        df.loc[:, ['photo_id', 'face_id']] = df.loc[:, ['photo_id', 'face_id']].astype('int')
        df.sort_values(by=['photo_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def __get_gradient(self, image, window_half_height, stride, image_height):
        window_height = window_half_height * 2
        i = 0
        features = []
        while(i + window_height < image_height):
            window_center = i + window_half_height
            feature = np.sum(np.abs(image[i:window_center, :] - image[window_center: i + window_height, :]))
            features.append(feature)
            i += stride
        
        return features
    
    def generate_representation(self, image_path, representation_path, window_half_height=4, image_height=80):
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)   
        features = self.__get_gradient(image, window_half_height, self.stride, image_height)
        
        plt.clf()
        plt.plot(features)
        plt.savefig(representation_path + '/gradient_representation.png')
