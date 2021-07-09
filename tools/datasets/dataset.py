from abc import ABC, abstractmethod

# This is an ABC: Abstract base class
class Dataset(ABC):
    
    def init_dataset(self):
        pass

    def parse_dataset(self):
        pass