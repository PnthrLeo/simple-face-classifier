import math
import dearpygui.dearpygui as dpg
from ._controller import Controller
import models
import numpy as np


class BestTTSizeSearchController(Controller):
    def __init__(self, width, height):
        super().__init__(width=width, height=height)
        self.image_folder_path = None
    
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
        mode = dpg.get_value('mode_combo_list')
        if mode == 'test_size':
            cls = self.__get_classifier_class()
            test_sizes = list(range(1, 120))
            
            param = dpg.get_value('method_input')   
            clf = cls(param)
            
            acc_score_list = []
            for idx, n in enumerate(test_sizes):
                if idx == 0:
                    _, acc_score, _ = clf.run_test_first_n(self.image_folder_path, n, True)
                else:
                    _, acc_score, _ = clf.run_test_first_n(self.image_folder_path, n, False)
                acc_score_list.append(acc_score)
                dpg.set_value('progress_bar', (idx + 1) / len(test_sizes))

            dpg.set_item_label('plot_label', 'Accuracy vs images quantity')
            dpg.set_item_label('plot_x_axis', 'images quantity')
            dpg.set_item_label('plot_y_axis', 'accuracy')
            dpg.set_axis_limits('plot_x_axis', 0, len(test_sizes))
            dpg.set_axis_limits('plot_y_axis', 0, 1.1)
            dpg.set_value('plot_series', [test_sizes, acc_score_list])
        elif mode=='train_size':
            cls = self.__get_classifier_class()
            train_sizes = list(range(1, 10))
            
            param = dpg.get_value('method_input')   
            clf = cls(param)
            
            acc_score_list = []
            for idx, n in enumerate(train_sizes):
                _, acc_score, _ = clf.run_train_first_n(self.image_folder_path, n)
                acc_score_list.append(acc_score)
                dpg.set_value('progress_bar', (idx + 1) / len(train_sizes))
            
            dpg.set_item_label('plot_label', 'Accuracy vs train size')
            dpg.set_item_label('plot_x_axis', 'train size')
            dpg.set_item_label('plot_y_axis', 'accuracy')
            dpg.set_axis_limits('plot_x_axis', 0, 11)
            dpg.set_axis_limits('plot_y_axis', 0, 1.1)
            dpg.set_value('plot_series', [train_sizes, acc_score_list])
        elif mode=='train_test_split':
            cls = self.__get_classifier_class()
            test_sizes = np.arange(0.1, 1.0, 0.1)
            
            param = dpg.get_value('method_input')
            
            acc_score_list = []
            for idx, test_size in enumerate(test_sizes):
                clf = cls(param)
                acc_score, _ = clf.run_cv(self.image_folder_path, test_size=test_size)
                acc_score_list.append(acc_score)
                dpg.set_value('progress_bar', (idx + 1) / len(test_sizes))
            
            dpg.set_item_label('plot_label', 'Accuracy vs test size')
            dpg.set_item_label('plot_x_axis', 'test size')
            dpg.set_item_label('plot_y_axis', 'accuracy')
            dpg.set_axis_limits('plot_x_axis', 0, 1.0)
            dpg.set_axis_limits('plot_y_axis', 0, 1.1)
            dpg.set_value('plot_series', [test_sizes, acc_score_list])
                
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
    
    def select_train_dataset(self, sender, app_data):
        dpg.show_item('select_data_dialog')
        dpg.focus_item('select_data_dialog')
    
    def select_folder(self, sender, app_data):
        dpg.set_value('data_path_text', f'Path: {app_data["current_path"]}')
        self.image_folder_path = app_data["current_path"]

    def delete(self):
        dpg.delete_item('control_window')
        dpg.delete_item('plot_window')
        dpg.delete_item('select_data_dialog')
        dpg.delete_item('plot_theme')
