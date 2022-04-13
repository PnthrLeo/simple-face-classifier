from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit


class Classifier(ABC):
    @abstractmethod
    def __init__(self, parameter):
        pass
    
    @abstractmethod
    def calculate_features(self, image_folder_path):
        pass
    
    def run_split(self, image_folder_path, test_size=0.25, random_state=42):
        df = self.calculate_features(image_folder_path)
        y = df['face_id']
        X = df.drop(['photo_id', 'face_id'], axis=1)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_index, test_index = next(sss.split(X, y))

        self.df_train = df.iloc[train_index]
        self.df_test = df.iloc[test_index]

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        self.fit(X_train, y_train)
        
        y_pred = self.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        false_predictions = y_pred != y_test
        
        return y_pred, acc_score, false_predictions
    
    def run_cv(self, image_folder_path, n_splits=10, test_size=0.25, random_state=42):
        df = self.calculate_features(image_folder_path)
        y = df['face_id']
        X = df.drop(['photo_id', 'face_id'], axis=1)
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        
        acc_score_list = []
        
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred)
            acc_score_list.append(acc_score)
        
        acc_score_mean = np.mean(np.array(acc_score_list))
        return acc_score_mean, acc_score_list
    
    def run_manual(self, image_folder_path_train, image_folder_path_test):
        self.df_train = self.calculate_features(image_folder_path_train)
        self.df_test = self.calculate_features(image_folder_path_test)
        
        y_train = self.df_train['face_id']
        X_train = self.df_train.drop(['photo_id', 'face_id'], axis=1)
        y_test = self.df_test['face_id']
        X_test = self.df_test.drop(['photo_id', 'face_id'], axis=1)
        
        self.fit(X_train, y_train)
        
        y_pred = self.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        false_predictions = y_pred != y_test
        
        return y_pred, acc_score, false_predictions

    def fit(self, train_X, train_y):
        return self.clf.fit(train_X, train_y)
    
    def predict(self, test_X):
        return self.clf.predict(test_X)
