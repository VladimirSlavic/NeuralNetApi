class AbstractLoss:

    def loss(self, x, y):
        raise NotImplementedError('abstract loss cannot be used')
