import dearpygui.dearpygui as dpg
from ._controller import Controller
import models
import numpy as np


class ParameterSearchController(Controller):
    def __init__(self, width, height):
        super().__init__(width=width, height=height)
        self.image_folder_path = None
        
    def run_test(self, sender, app_data):
        dpg.set_value('best_parameter_text', 'Best Parameter: None')
        dpg.set_value('best_accuracy_text', f'Best Accruracy: None')
        
        cls, plot_name, x_axis_name, y_axis_name = self.__get_classifier_class()
        params_grid = None
        
        method = dpg.get_value('method_combo_list')
        match method:
            case 'histogram':
                params_grid = list(range(1, 256))
            case 'dft':
                params_grid = list(range(1, 30))
            case 'dct':
                params_grid = list(range(1, 70))
            case 'scale':
                params_grid = list(range(10, 50))
            case 'gradient':
                params_grid = list(range(1, 20))
        
        acc_score_list = []
        for idx, param in enumerate(params_grid):
            clf = cls(param)
            acc_score, _ = clf.run_cv(self.image_folder_path)
            acc_score_list.append(acc_score)
            dpg.set_value('progress_bar', (idx + 1) / len(params_grid))
        
        dpg.set_item_label('plot_label', plot_name)
        dpg.set_item_label('plot_x_axis', x_axis_name)
        dpg.set_item_label('plot_y_axis', y_axis_name)
        dpg.set_axis_limits('plot_x_axis', min(params_grid), max(params_grid))
        dpg.set_axis_limits('plot_y_axis', 0, 1.1)
        dpg.set_value('plot_series', [params_grid, acc_score_list])
        
        dpg.set_value('best_parameter_text', f'Best Parameter: {params_grid[(np.argmax(acc_score_list))]}')
        dpg.set_value('best_accuracy_text', f'Best Accruracy: {np.max(acc_score_list)}')
        
    def __get_classifier_class(self):
        method = dpg.get_value('method_combo_list')
        clf = None
        plot_name = None
        
        match method:
            case 'histogram':
                cls = models.HistogramClassifier
                plot_name = 'Accuracy vs bins'
                x_axis_name = 'bins'
                y_axis_name = 'accuracy'
            case 'dft':
                cls = models.DFTClassifier
                plot_name = 'Accuracy vs l1_radius'
                x_axis_name = 'l1_radius'
                y_axis_name = 'accuracy'
            case 'dct':
                cls = models.DCTClassifier
                plot_name = 'Accuracy vs l1_radius_quarter'
                x_axis_name = 'l1_radius_quarter'
                y_axis_name = 'accuracy'
            case 'scale':
                cls = models.ScaleClassifier
                plot_name = 'Accuracy vs scale_percent'
                x_axis_name = 'scale_percent'
                y_axis_name = 'accuracy'
            case 'gradient':
                cls = models.GradientClassifier
                plot_name = 'Accuracy vs stride'
                x_axis_name = 'stride'
                y_axis_name = 'accuracy'

        return cls, plot_name, x_axis_name, y_axis_name

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
