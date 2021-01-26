import numpy as np
from numba import jitclass
from numba import int64


queue_spec = [
    ('capacity', int64),
    ('head', int64),
    ('tail', int64),
    ('values', int64[:]),
]


@jitclass(queue_spec)
class Queue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.head = self.tail = 0
        self.values = np.empty(capacity, dtype=np.int64)

    def resize(self, new_capacity):
        self.capacity = new_capacity
        self.tail = min(self.tail, new_capacity)

        new_values = np.empty(new_capacity, dtype=np.int64)
        new_values[:self.tail] = self.values[:self.tail]
        self.values = new_values

    def push(self, value):
        if self.tail == self.capacity:
            self.resize(2 * self.capacity)

        self.values[self.tail] = value
        self.tail += 1

    def pop(self):
        if self.head < self.tail:
            self.head += 1
            return self.values[self.head - 1]

        return -1

    def empty(self):
        return self.head == self.tail

    @property
    def size(self):
        return self.tail - self.head
