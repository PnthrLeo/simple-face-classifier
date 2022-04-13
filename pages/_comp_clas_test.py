import dearpygui.dearpygui as dpg
from controllers import CompClasTestController
from ._page import Page


_CONTROL_WINDOW_WIDTH = 300
_COMBO_WIDTH = 125


class CompClasTestPage(Page):
    def __init__(self, width, height):
        self.control_window_width = _CONTROL_WINDOW_WIDTH
        self.control_window_height = height
        self.media_window_width = min(width-_CONTROL_WINDOW_WIDTH, height) / 3
        self.media_window_height = self.media_window_width
        self.explorer_window_height = height / 2
        self.explorer_window_width = width / 2

        self.controller = CompClasTestController(width, height)
        
        self.__draw_control_window_elements()
        self.__draw_photo_window_elements()
        self.__draw_histogram_window_elements()
        self.__draw_dft_window_elements()
        self.__draw_dct_window_elements()
        self.__draw_scale_window_elements()
        self.__draw_gradient_window_elements()
        self.__draw_result_window_elements()
        self.__draw_data_explorer_window_elements(show=False)
    
    def __draw_control_window_elements(self):
        with dpg.window(label='Control', width=self.control_window_width, height=self.control_window_height, pos=(0, 0), tag='control_window'):
            with dpg.menu_bar():
                with dpg.menu(label='Choose Program Mode', tag='program_mode_menu'):
                    dpg.add_menu_item(label='train-train Test', callback=self.controller.switch_page, tag='train_train_test_ref')
                    dpg.add_menu_item(label='Parameter Search', callback=self.controller.switch_page,  tag='parameter_search_ref')
                    dpg.add_menu_item(label='Best train-test Size Search', callback=self.controller.switch_page, tag='best_tt_size_search_ref')
                    dpg.add_menu_item(label='Bad Image Search', callback=self.controller.switch_page, tag='bad_im_search_ref')
                    dpg.add_menu_item(label='Composed Classifier Test', callback=self.controller.switch_page, tag='comp_clas_test_ref')

            dpg.add_button(label='Select Train Dataset', callback=self.controller.select_dataset, tag='select_train_data_button')
            dpg.add_text(default_value='Path:', tag='train_data_path_text')
            dpg.add_button(label='Select Test Dataset', callback=self.controller.select_dataset, tag='select_test_data_button')
            dpg.add_text(default_value='Path:', tag='test_data_path_text')
            dpg.add_input_int(label='Number of Bins (histogram)', width=125, min_value=1, max_value=255, min_clamped=True, max_clamped=True, default_value=1, tag='histogram_input')
            dpg.add_input_int(label='L1 Radius (dft)', width=125, min_value=1, max_value=255, min_clamped=True, max_clamped=True, default_value=1, tag='dft_input')
            dpg.add_input_int(label='L1 Radius Quarter (dct)', width=125, min_value=1, max_value=255, min_clamped=True, max_clamped=True, default_value=1, tag='dct_input')
            dpg.add_input_int(label='Scale Percentage (scale)', width=125, min_value=1, max_value=255, min_clamped=True, max_clamped=True, default_value=1, tag='scale_input')
            dpg.add_input_int(label='Stride (gradient)', width=125, min_value=1, max_value=255, min_clamped=True, max_clamped=True, default_value=1, tag='gradient_input')
            dpg.add_button(label='Run Test', callback=self.controller.run_test, tag='run_test_button')
            dpg.add_text(default_value='Accuracy: None', tag='clf_accuracy')
            dpg.add_input_int(label='Select Test Photo', width=125, min_value=1, min_clamped=True, default_value=1, enabled=False, callback=self.controller.select_test_photo, tag='photo_input')

    def __draw_photo_window_elements(self):
        with dpg.window(label='Photo', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH, 0), tag='photo_window'):
            pass
    
    def __draw_histogram_window_elements(self):
        with dpg.window(label='Histogram Feature', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH  + self.media_window_width, 0), tag='histogram_window'):
            pass
    
    def __draw_dft_window_elements(self):
        with dpg.window(label='DFT Feature', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH  + 2*self.media_window_width, 0), tag='dft_window'):
            pass

    def __draw_dct_window_elements(self):
        with dpg.window(label='DCT Feature', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH, self.media_window_height), tag='dct_window'):
         pass
    
    def __draw_scale_window_elements(self):
        with dpg.window(label='Scale Feature', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH + self.media_window_width, self.media_window_height), tag='scale_window'):
         pass

    def __draw_gradient_window_elements(self):
        with dpg.window(label='Gradient Feature', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH + 2*self.media_window_width, self.media_window_height), tag='gradient_window'):
         pass

    def __draw_result_window_elements(self):
        with dpg.window(label='Result', width=self.media_window_width, height=self.media_window_height, pos=(_CONTROL_WINDOW_WIDTH, 2*self.media_window_height), tag='result_window'):
            pass
    
    def __draw_data_explorer_window_elements(self, show):
        dpg.add_file_dialog(height=self.explorer_window_height,
                            width=self.explorer_window_width,
                            directory_selector=True, show=show,
                            tag='select_data_dialog')
