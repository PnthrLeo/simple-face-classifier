import dearpygui.dearpygui as dpg
from pages import TrainTrainPage

WIDTH = 1000
HEIGHT = 760

dpg.create_context()

TrainTrainPage(width=WIDTH, height=HEIGHT)

dpg.create_viewport(title='Face Cloassificator', width=WIDTH, height=HEIGHT)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
