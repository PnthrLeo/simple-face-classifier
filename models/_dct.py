from pathlib import Path

import pandas as pd
import numpy as np
import cv2 as cv
from scipy.fftpack import dct, idct
from skimage.io import imread, imsave
from sklearn.neighbors import KNeighborsClassifier

from ._classifier import Classifier


class DCTClassifier(Classifier):
    def __init__(self, l1_radius_quarter):
        self.l1_radius_quarter = l1_radius_quarter
        self.clf =  KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        self.df_train = None
        self.df_test = None
    
    def calculate_features(self, image_folder_path):
        folder_path = Path(image_folder_path)
        df = pd.DataFrame(columns=['photo_id', 'face_id', *list(range(1, (self.l1_radius_quarter) ** 2 + 1))])
        
        for image_path in folder_path.glob('*.jpg'):
            image = imread(image_path, as_gray=True) 
            image_dct = self.__dct2(image)
            features = image_dct[0:self.l1_radius_quarter, 0:self.l1_radius_quarter]
            
            photo_id, face_id = map(int, image_path.stem.split('_'))
            row = [photo_id, face_id, *features.ravel()]
            df = pd.concat([df, pd.DataFrame([row], columns=df.columns)])

        df = df.astype('float64')
        df.loc[:, ['photo_id', 'face_id']] = df.loc[:, ['photo_id', 'face_id']].astype('int')
        df.sort_values(by=['photo_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def generate_representation(self, image_path, representation_path):
        image = imread(image_path, as_gray=True) 
        image_dct = self.__dct2(image)
        features = image_dct[0:self.l1_radius_quarter, 0:self.l1_radius_quarter]
        features = 20*np.log(features)
        cv.imwrite(representation_path + '/dct_representation.png', features)
    
    def __dct2(self, a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    def __idct2(self, a):
        return idct(idct(a.T, norm='ortho').T, norm='ortho')
