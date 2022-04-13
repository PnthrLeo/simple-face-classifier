from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from ._classifier import Classifier


class DFTClassifier(Classifier):
    def __init__(self, l1_radius):
        self.l1_radius = l1_radius
        self.clf =  KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        self.df_train = None
        self.df_test = None
    
    def calculate_features(self, image_folder_path):
        folder_path = Path(image_folder_path)
        df = pd.DataFrame(columns=['photo_id', 'face_id', *list(range(1, (self.l1_radius * 2) ** 2 + 1))])
        for image_path in folder_path.glob('*.jpg'):
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            dft = np.fft.fft2(image)
            dft_shift = np.fft.fftshift(dft)

            x_center, y_center = dft_shift.shape
            x_center, y_center = x_center // 2, y_center // 2
            
            features = dft_shift[y_center - self.l1_radius: y_center + self.l1_radius,
                                x_center - self.l1_radius: x_center + self.l1_radius]
            features = np.abs(features)
            
            photo_id, face_id = map(int, image_path.stem.split('_'))
            row = [photo_id, face_id, *features.ravel()]
            df = pd.concat([df, pd.DataFrame([row], columns=df.columns)])
        
        df = df.astype('float64')
        df.loc[:, ['photo_id', 'face_id']] = df.loc[:, ['photo_id', 'face_id']].astype('int')
        df.sort_values(by=['photo_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def generate_representation(self, image_path, representation_path):
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        dft = np.fft.fft2(image)
        dft_shift = np.fft.fftshift(dft)

        x_center, y_center = dft_shift.shape
        x_center, y_center = x_center // 2, y_center // 2
        
        features = dft_shift[y_center - self.l1_radius: y_center + self.l1_radius,
                            x_center - self.l1_radius: x_center + self.l1_radius]
        features = np.abs(features)
        features = 20*np.log(features)
        cv.imwrite(representation_path + '/dft_representation.png', features)
