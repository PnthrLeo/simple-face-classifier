import dearpygui.dearpygui as dpg
from controllers import ParameterSearchController
from ._page import Page


_CONTROL_WINDOW_WIDTH = 300
_COMBO_WIDTH = 125


class ParameterSearchPage(Page):
    def __init__(self, width, height):
        self.control_window_width = _CONTROL_WINDOW_WIDTH
        self.control_window_height = height
        self.plot_window_width = min(width-_CONTROL_WINDOW_WIDTH, height)
        self.plot_window_height = self.plot_window_width
        self.explorer_window_height = height / 2
        self.explorer_window_width = width / 2

        self.controller = ParameterSearchController(width, height)
        
        self.__draw_control_window_elements()
        self.__draw_plot_window_elements()
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

            dpg.add_button(label='Select Train Dataset', callback=self.controller.select_train_dataset, tag='select_data_button')
            dpg.add_text(default_value='Path:', tag='data_path_text')
            dpg.add_combo(label='Recognition Method', items=['histogram', 'dft', 'dct', 'scale', 'gradient'],
                            width=_COMBO_WIDTH, default_value='histogram', tag='method_combo_list')
            dpg.add_button(label='Run Test', callback=self.controller.run_test, tag='run_test_button')
            dpg.add_progress_bar(default_value=0.0, width=_COMBO_WIDTH, tag='progress_bar')
            dpg.add_text(default_value='Best Parameter: None', tag='best_parameter_text')
            dpg.add_text(default_value='Best Accruracy: None', tag='best_accuracy_text')

    def __draw_plot_window_elements(self):
        with dpg.theme(tag='plot_theme'):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 10, category=dpg.mvThemeCat_Plots)
        
        with dpg.window(label='Plot', width=self.plot_window_width, height=self.plot_window_height, pos=(_CONTROL_WINDOW_WIDTH, 0), tag='plot_window'):
                with dpg.plot(label='...', width=self.plot_window_width-30, height=self.plot_window_height-30, tag='plot_label'):
                    dpg.add_plot_legend()

                    dpg.add_plot_axis(dpg.mvXAxis, tag='plot_x_axis')
                    dpg.add_plot_axis(dpg.mvYAxis, tag='plot_y_axis')

                    dpg.add_line_series([], [], parent='plot_y_axis', tag='plot_series')
                    
                    dpg.bind_item_theme('plot_series', 'plot_theme')
    
    def __draw_explorer_window_elements(self, show):
        dpg.add_file_dialog(height=self.explorer_window_height,
                            width=self.explorer_window_width,
                            directory_selector=True, show=show,
                            callback=self.controller.select_folder,
                            tag='select_data_dialog')
