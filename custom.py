from PIL import Image


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


class Lambda_scheduler():
    """
    Parameters:
    * epochs_amount - the amount of epoch in a whole learning session

    * decay_start - the amount of epochs after which the scheduler will start working

    * offset (optional, 0 by default) - amount of epochs, that have passed in the previous session
    """

    def __init__(self, epochs_amount, decay_start, offset=0):
        self.offset = offset
        self.epchs_amount = epochs_amount
        self.decay_start = decay_start

    def step(self, epoch):
        return 1.0 - max(0, epoch+self.offset - self.decay_start)/(self.epchs_amount - self.decay_start)


def to_rgb(image):
    rgb_img = Image.new('RGB', image.size)
    rgb_img.paste(image)
    return rgb_img
