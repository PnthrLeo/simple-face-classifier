from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import pandas as pd
import shutil


def run_split(image_folder_path, distination_path, test_size=0.3, random_state=42):
    df = generate_df(image_folder_path)
    y = df['face_id']
    X = df.drop(['photo_id', 'face_id'], axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(sss.split(X, y))

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]
    
    save_dataset(df_train, distination_path + '/train')
    save_dataset(df_test, distination_path + '/test')
    

def generate_df(image_folder_path):
    folder_path = Path(image_folder_path)
    df = pd.DataFrame(columns=['photo_id', 'face_id', 'image_path'])
    
    for image_path in folder_path.glob('*.jpg'):
        photo_id, face_id = map(int, image_path.stem.split('_'))
        row = [photo_id, face_id, image_path]
        df = pd.concat([df, pd.DataFrame([row], columns=df.columns)])
    
    df.loc[:, ['photo_id', 'face_id']] = df.loc[:, ['photo_id', 'face_id']].astype('int')
    df.sort_values(by=['photo_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def save_dataset(df, distination_path):
    for image_path in df['image_path']:
        p = Path(image_path)
        new_p = Path(distination_path) / p.name
        shutil.copy2(p.resolve(), new_p.resolve())

if __name__ == '__main__':
    run_split('./data/orl/', './data/orl_splitted/')
