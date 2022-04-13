from pathlib import Path

import cv2 as cv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from ._classifier import Classifier


class HistogramClassifier(Classifier):
    def __init__(self, bins):
        self.bins = bins
        self.clf =  KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        self.df_train = None
        self.df_test = None
    
    def calculate_features(self, image_folder_path):
        folder_path = Path(image_folder_path)
        df = pd.DataFrame(columns=['photo_id', 'face_id', *list(range(1, self.bins + 1))])
        
        for image_path in folder_path.glob('*.jpg'):
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            histr = cv.calcHist([image], [0], None, [self.bins],[0, 256])
            
            photo_id, face_id = map(int, image_path.stem.split('_'))
            row = [photo_id, face_id, *histr.ravel()]
            df = pd.concat([df, pd.DataFrame([row], columns=df.columns)])
        
        df = df.astype('float64')
        df.loc[:, ['photo_id', 'face_id']] = df.loc[:, ['photo_id', 'face_id']].astype('int')
        df.sort_values(by=['photo_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def generate_representation(self, image_path, representation_path):
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        histr = cv.calcHist([image], [0], None, [self.bins],[0, 256])
        
        plt.clf()
        plt.plot(histr)
        plt.savefig(representation_path + '/histogram_representation.png')
