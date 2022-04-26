import dearpygui.dearpygui as dpg
from controllers import BadImSearchController
from ._page import Page


_CONTROL_WINDOW_WIDTH = 300
_COMBO_WIDTH = 125


class BadImSearchPage(Page):
    def __init__(self, width, height):
        self.control_window_width = _CONTROL_WINDOW_WIDTH
        self.control_window_height = height
        self.media_window_width = min(width-_CONTROL_WINDOW_WIDTH, height) / 2
        self.media_window_height = self.media_window_width
        self.explorer_window_height = height / 2
        self.explorer_window_width = width / 2

        self.controller = BadImSearchController(width, height)
        
        self.__draw_control_window_elements()
        self.__draw_photo_window_elements()
        self.__draw_photo_feature_window_elements()
        self.__draw_result_window_elements()
        self.__draw_result_feature_window_elements()
        self.__draw_explorer_window_elements(show=False)
    
    def __draw_control_window_elements(self):
        with dpg.window(label='Control', width=self.control_window_width, height=self.control_window_height, pos=(0, 0), tag='control_window'):
            with dpg.menu_bar():
                with dpg.menu(label='Choose Program Mode', tag='program_mode_menu'):
                    dpg.add_menu_item(label='train-train Test', callback=self.controller.switch_page, tag='train_train_test_ref')
                    dpg.add_menu_item(label='Parameter Search', callback=self.controller.switch_page,  tag='parameter_search_ref')
                    dpg.add_menu_item(label='Train Test tests', callback=self.controller.switch_page, tag='train_test_tests_ref')
                    dpg.add_menu_item(label='Bad Image Search', callback=self.controller.switch_page, tag='bad_im_search_ref')
                    dpg.add_menu_item(label='Composed Classifier Test', callback=self.controller.switch_page, tag='comp_clas_test_ref')

            dpg.add_button(label='Select Train Dataset', callback=self.controller.select_dataset, tag='select_train_data_button')
            dpg.add_text(default_value='Path:', tag='train_data_path_text')
            dpg.add_button(label='Select Test Dataset', callback=self.controller.select_dataset, tag='select_test_data_button')
            dpg.add_text(default_value='Path:', tag='test_data_path_text')
            dpg.add_combo(label='Recognition Method', items=['histogram', 'dft', 'dct', 'scale', 'gradient'],
                            width=_COMBO_WIDTH, default_value='histogram', callback=self.controller.switch_classificator_features, tag='method_combo_list')
            dpg.add_input_int(label='Number of Bins', width=125, min_value=1, max_value=255, min_clamped=True, max_clamped=True, default_value=1, tag='method_input')
            dpg.add_input_int(label='Select Bad Photo', width=125, min_value=1, min_clamped=True, default_value=1, enabled=False, callback=self.controller.select_bad_photo, tag='photo_input')
            dpg.add_button(label='Run Test', callback=self.controller.run_test, tag='run_test_button')
            dpg.add_combo(label='Mode', items=['all_images', 'bad_images'], width=_COMBO_WIDTH, default_value='all_images', callback=self.controller.switch_mode, tag='image_combo_mode')

    def __draw_photo_window_elements(self):
        with dpg.window(label='Photo', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH, 0), tag='photo_window'):
            pass
    
    def __draw_photo_feature_window_elements(self):
        with dpg.window(label='Photo Feature', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH + self.media_window_width, 0), tag='photo_feature_window'):
            pass
    
    def __draw_result_window_elements(self):
        with dpg.window(label='Result', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH, self.media_window_height), tag='result_window'):
            pass
    
    def __draw_result_feature_window_elements(self):
        with dpg.window(label='Result Feature', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH + self.media_window_width, self.media_window_height), tag='result_feature_window'):
            pass
    
    def __draw_explorer_window_elements(self, show):
        dpg.add_file_dialog(height=self.explorer_window_height,
                            width=self.explorer_window_width,
                            directory_selector=True, show=show,
                            tag='select_data_dialog')
