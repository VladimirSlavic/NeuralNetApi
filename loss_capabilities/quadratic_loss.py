from loss_capabilities.losses import AbstractLoss
import numpy as np
import math


class QuadraticLoss(AbstractLoss):
    def __init__(self, one_hot=True):
        self.one_hot = one_hot
        self.name = 'quadratic'
        self.counter = 0

    def get_name(self):
        return self.name

    def is_one_hot(self):
        return self.one_hot

    def loss(self, x, y):
        N = len(x)
        if not self.one_hot:
            y_one_hot = np.zeros((y.size, x.shape[1]))
            y_one_hot[np.arange(y.size), y] = 1
            y = y_one_hot

        loss = 0.5 * np.linalg.norm(x - y) ** 2
        loss /= N
        dx = (x - y)
        dx /= N
        return loss, dx
