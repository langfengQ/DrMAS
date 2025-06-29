from typing import List, Dict, Any

class SimpleMemory:
    """
    Memory manager: responsible for storing & fetching per-environment history records.
    """
    def __init__(self):
        self._data = None

    # ---------- basic list-like behaviour ----------
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    # ---------- public API ----------
    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]

    def store(self, record: Dict[str, List[Any]]):
        keys = list(record.keys())
        batch_size = len(record[keys[0]])
        for i in range(batch_size):
            self._data[i].append({k: record[k][i] for k in keys})