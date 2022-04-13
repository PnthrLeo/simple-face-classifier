from abc import ABC, abstractmethod


class Page(ABC):
    @abstractmethod
    def __init__(self, width, height):
        pass
