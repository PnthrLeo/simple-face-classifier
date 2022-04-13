import dearpygui.dearpygui as dpg
from ._controller import Controller
import models
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path


class CompClasTestController(Controller):
    def __init__(self, width, height):
        super().__init__(width=width, height=height)
        self.image_folder_path_train = None
        self.image_folder_path_test = None
        self.clfs = []

    def run_test(self, sender, app_data):
        self.clfs = [
            models.HistogramClassifier(dpg.get_value('histogram_input')),
            models.DFTClassifier(dpg.get_value('dft_input')),
            models.DCTClassifier(dpg.get_value('dct_input')),
            models.ScaleClassifier(dpg.get_value('scale_input')),
            models.GradientClassifier(dpg.get_value('gradient_input'))
        ]
        
        y_pred_mat = None
        for clf in self.clfs:
            y_pred, _, _ = clf.run_manual(self.image_folder_path_train, self.image_folder_path_test)
            y_pred = y_pred.reshape(-1, 1)
            if y_pred_mat is None:
                y_pred_mat = y_pred
            else:
                y_pred_mat = np.concatenate((y_pred_mat, y_pred), axis=1)
        
        y_pred_comb = np.zeros(y_pred_mat.shape[0])
        for i in range(y_pred_mat.shape[0]):
            unique, counts = np.unique(y_pred_mat[i], return_counts=True)
            y_pred_comb[i] = unique[np.argmax(counts)]
        
        acc_score = accuracy_score(self.clfs[0].df_test['face_id'], y_pred_comb.astype(int))
        dpg.set_value('clf_accuracy', f'Accuracy: {acc_score}')
        
        self.df_test = self.clfs[0].df_test
        self.df_pred = y_pred_comb
        
        dpg.configure_item('photo_input', enabled=True, min_value=1, max_value=len(self.df_test))
        
        self.__show_instance(0)
    
    def select_test_photo(self, sender, app_data):
        idx = app_data
        self.__show_instance(idx - 1)
    
    def __show_instance(self, idx):
        image_row = self.df_test.iloc[idx]
        
        image_path = Path(self.image_folder_path_test)
        image_path = image_path / f'{int(image_row["photo_id"])}_{int(image_row["face_id"])}.jpg'
        
        self.__draw_image(str(image_path), 'photo')
        
        for clf in self.clfs:
            clf.generate_representation(image_path, './__app_cache__')
        
        features=['histogram', 'dft', 'dct', 'scale', 'gradient']
        
        for feature in features:
            feature_path = f'./__app_cache__/{feature}_representation.png'
            self.__draw_image(feature_path, feature)
        
        result_path = f'./data/orl/{int((self.df_pred[idx]-1)*10 + 1)}_{int(self.df_pred[idx])}.jpg'
        self.__draw_image(result_path, 'result')

    def __draw_image(self, image_path, window_name):
        if dpg.does_item_exist(f'{window_name}_image'):
            dpg.delete_item(f'{window_name}_image')
        if dpg.does_item_exist(f'{window_name}_drawlist'):
            dpg.delete_item(f'{window_name}_drawlist')
        
        width, height, _, data = dpg.load_image(image_path)
        
        with dpg.texture_registry():
            dpg.add_static_texture(width, height, data, tag=f'{window_name}_image')

        width = dpg.get_item_configuration(f'{window_name}_window')['width']
        height = dpg.get_item_configuration(f'{window_name}_window')['height']
        
        with dpg.drawlist(width=width-15, height=height-15, tag=f'{window_name}_drawlist', parent=f'{window_name}_window'):
            dpg.draw_image(f'{window_name}_image', (0, 0), (width-15, height-15), uv_min=(0, 0), uv_max=(1, 1))

    def select_dataset(self, sender, app_data):
        match sender:
            case 'select_train_data_button':
                dpg.configure_item('select_data_dialog', callback=self.select_train_folder)
            case 'select_test_data_button':
                dpg.configure_item('select_data_dialog', callback=self.select_test_folder)
        dpg.show_item('select_data_dialog')
        dpg.focus_item('select_data_dialog')
    
    def select_train_folder(self, sender, app_data):
        dpg.set_value('train_data_path_text', f'Path: {app_data["current_path"]}')
        self.image_folder_path_train = app_data["current_path"]
    
    def select_test_folder(self, sender, app_data):
        dpg.set_value('test_data_path_text', f'Path: {app_data["current_path"]}')
        self.image_folder_path_test = app_data["current_path"]
    
    def delete(self):
        dpg.delete_item('control_window')
        dpg.delete_item('photo_window')
        dpg.delete_item('histogram_window')
        dpg.delete_item('dft_window')
        dpg.delete_item('dct_window')
        dpg.delete_item('scale_window')
        dpg.delete_item('gradient_window')
        dpg.delete_item('result_window')
        dpg.delete_item('select_data_dialog')
