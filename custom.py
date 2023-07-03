class Buffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.elems = []

    def append(self, elem):
        if len(self.elems) >= self.capacity:
            self._pop()
        self.elems.append(elem)

    def _pop(self):
        self.elems.pop(0)

    def __str__(self) -> str:
        return str(self.elems)

    def __len__(self):
        return len(self.elems)
