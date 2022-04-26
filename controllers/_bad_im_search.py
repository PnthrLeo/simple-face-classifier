import dearpygui.dearpygui as dpg
from ._controller import Controller
import models
import pandas as pd
from pathlib import Path


class BadImSearchController(Controller):
    def __init__(self, width, height):
        super().__init__(width=width, height=height)
        self.image_folder_path_train = None
        self.image_folder_path_test = None
        self.images_df = None
        self.pred = None
        self.bad_images_df = None
        self.bad_pred = None
        self.clf = None

    def switch_classificator_features(self, sender, app_data):
        method = app_data
        label = None
        
        match method:
            case 'histogram':
                label = 'Number of Bins'
            case 'dft':
                label = 'L1 Radius'
            case 'dct':
                label = 'L1 Radius Quarter'
            case 'scale':
                label = 'Scale Percentage'
            case 'gradient':
                label = 'Stride'
        
        dpg.set_item_label('method_input', label)
        
    def run_test(self, sender, app_data):
        cls = self.__get_classifier_class()
        
        param = dpg.get_value('method_input')
        
        self.clf = cls(param)
        y_pred, _, false_predictions = self.clf.run_manual(self.image_folder_path_train, self.image_folder_path_test)
        
        self.images_df = self.clf.df_test
        self.pred = y_pred
        self.bad_images_df = self.clf.df_test[pd.Series(false_predictions, dtype=bool).values]
        self.bad_pred = y_pred[pd.Series(false_predictions, dtype=bool).values]
        
        dpg.set_value('image_combo_mode', 'all_images')
        dpg.configure_item('photo_input', enabled=True, min_value=1, max_value=len(self.images_df))
        dpg.set_value('photo_input', 1)
        
        self.__show_instance(0)
    
    def __show_instance(self, idx):
        df = None
        pred = None
        
        match dpg.get_value('image_combo_mode'):
            case 'all_images':
                df = self.images_df
                pred = self.pred
            case 'bad_images':
                df = self.bad_images_df
                pred = self.bad_pred
        
        image_row = df.iloc[idx]
        
        image_path = Path(self.image_folder_path_test)
        image_path = image_path / f'{int(image_row["photo_id"])}_{int(image_row["face_id"])}.jpg'
        
        self.clf.generate_representation(image_path, './__app_cache__')
        
        photo_path = str(image_path)
        feature_path = f'./__app_cache__/{dpg.get_value("method_combo_list")}_representation.png'
        result_path = f'./data/orl/{int((pred[idx]-1)*10 + 1)}_{int(pred[idx])}.jpg'

        self.__draw_image(photo_path, 'photo')
        self.__draw_image(feature_path, 'photo_feature')
        
        self.clf.generate_representation(result_path, './__app_cache__')
        self.__draw_image(result_path, 'result')
        self.__draw_image(feature_path, 'result_feature')
    
    def select_bad_photo(self, sender, app_data):
        idx = app_data
        self.__show_instance(idx - 1)

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
            
    def __get_classifier_class(self):
        method = dpg.get_value('method_combo_list')
        clf = None
        
        match method:
            case 'histogram':
                cls = models.HistogramClassifier
            case 'dft':
                cls = models.DFTClassifier
            case 'dct':
                cls = models.DCTClassifier
            case 'scale':
                cls = models.ScaleClassifier
            case 'gradient':
                cls = models.GradientClassifier

        return cls

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
        
    def switch_mode(self, sender, app_data):
        mode = app_data
        match app_data:
            case 'all_images':
                dpg.set_value('image_combo_mode', 'all_images')
                dpg.configure_item('photo_input', enabled=True, min_value=1, max_value=len(self.images_df))
                dpg.set_value('photo_input', 1)
            case 'bad_images':
                dpg.set_value('image_combo_mode', 'bad_images')
                dpg.configure_item('photo_input', enabled=True, min_value=1, max_value=len(self.bad_images_df))
                dpg.set_value('photo_input', 1)
        self.__show_instance(0)

    def delete(self):
        dpg.delete_item('control_window')
        dpg.delete_item('photo_window')
        dpg.delete_item('photo_feature_window')
        dpg.delete_item('result_window')
        dpg.delete_item('result_feature_window')
        dpg.delete_item('select_data_dialog')
