from abc import ABC, abstractmethod

# This is an ABC: Abstract base class
class Format(ABC):

    def init_output(self):
        pass
    
    def parse_label(self):
        pass

    def write_data(self):
        pass