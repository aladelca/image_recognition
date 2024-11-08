from abc import ABC, abstractmethod
from dataclasses import dataclass


class DataExtractInterface(ABC):
    @abstractmethod    
    def extract_data(self, query: str) -> str:
        pass