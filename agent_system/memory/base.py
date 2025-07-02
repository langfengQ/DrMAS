from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseMemory(ABC):
    """
    Base class for memory management. Defines the interface for memory modules.
    """

    @abstractmethod
    def __len__(self):
        """Return the number of memory slots."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """Access memory of specific environment index."""
        pass

    @abstractmethod
    def reset(self, batch_size: int):
        """
        Reset memory with given batch size.
        """
        pass

    @abstractmethod
    def store(self, record: Dict[str, List[Any]]):
        """
        Stores a new batch of records into memory.
        """
        pass

    @abstractmethod
    def fetch(self, step: int):
        """
        Fetches memory records at a specific time step across all environments.
        """
        pass