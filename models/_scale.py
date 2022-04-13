from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from ._classifier import Classifier


class ScaleClassifier(Classifier):
    def __init__(self, scale_percent):
        self.scale_percent = scale_percent
        self.clf =  KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        self.df_train = None
        self.df_test = None
    
    def calculate_features(self, image_folder_path, height=80, width=70):
        height_scaled = int(np.ceil(height * self.scale_percent / 100))
        width_scaled = int(np.ceil(width * self.scale_percent / 100))
        dim = (height_scaled, width_scaled)
        
        folder_path = Path(image_folder_path)
        df = pd.DataFrame(columns=['photo_id', 'face_id', *list(range(1, dim[0]*dim[1] + 1))])
        for image_path in folder_path.glob('*.jpg'):
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            
            image_resized = cv.resize(image, dim, interpolation = cv.INTER_CUBIC)
            
            photo_id, face_id = map(int, image_path.stem.split('_'))
            row = [photo_id, face_id, *image_resized.ravel()]
            df = pd.concat([df, pd.DataFrame([row], columns=df.columns)])

        df = df.astype('float64')
        df.loc[:, ['photo_id', 'face_id']] = df.loc[:, ['photo_id', 'face_id']].astype('int')
        df.sort_values(by=['photo_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def generate_representation(self, image_path, representation_path, height=80, width=70):
        height_scaled = int(np.ceil(height * self.scale_percent / 100))
        width_scaled = int(np.ceil(width * self.scale_percent / 100))
        dim = (height_scaled, width_scaled)
        
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        image_resized = cv.resize(image, dim, interpolation = cv.INTER_CUBIC)

        cv.imwrite(representation_path + '/scale_representation.png', image_resized)
