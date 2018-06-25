from loss_capabilities.losses import AbstractLoss
import numpy as np
import math


class SoftMax(AbstractLoss):
    def __init__(self, one_hot=True):
        self.one_hot = one_hot
        self.name = 'softmax'
        self.counter = 0

    def get_name(self):
        return self.name

    def is_one_hot(self):
        return self.one_hot

    def loss(self, x, y):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]

        if self.one_hot:
            y = np.argmax(y, axis=1)
        loss = -np.sum(log_probs[np.arange(N), y]) / N

        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx
