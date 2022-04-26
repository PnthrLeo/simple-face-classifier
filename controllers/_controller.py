from abc import ABC, abstractmethod
import pages
import dearpygui.dearpygui as dpg


class Controller(ABC):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def switch_page(self, sender, app_data):
        self.delete()
        match sender:
            case 'train_train_test_ref':
                pages.TrainTrainPage(self.width, self.height)
            case 'parameter_search_ref':
                pages.ParameterSearchPage(self.width, self.height)
            case 'train_test_tests_ref':
                pages.BestTTSizeSearchPage(self.width, self.height)
            case 'bad_im_search_ref':
                pages.BadImSearchPage(self.width, self.height)
            case 'comp_clas_test_ref':
                pages.CompClasTestPage(self.width, self.height)
    
    @abstractmethod
    def delete(self):
        pass
    