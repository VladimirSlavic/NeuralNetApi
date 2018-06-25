from neuralnet import optim
import numpy as np


class Controller:
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        num_val_examples = self.X_val.shape[0] if self.X_val is not None else None

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', self.X_train.shape[0])
        self.num_val_samples = kwargs.pop('num_val_samples', num_val_examples)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Too many arguments, didnt recognize %s' % extra)

        if not hasattr(optim, self.update_rule):
            raise ValueError('Update rule non existent "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}

        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def is_one_hot_vector(self):
        return self.model.loss_type.is_one_hot

    def accuracy(self, X, y, num_samples=None, batch_size=50):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N // batch_size
        if N % num_batches != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))

        if self.is_one_hot_vector():
            y = np.array(np.argmax(y, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    def predict(self, X):
        return self.model.loss(X)

    def train(self):

        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            new_epoch = (t + 1) % iterations_per_epoch == 0
            if new_epoch:
                self.epoch += 1

            first_iteration = (t == 0)
            last_iteration = (t == num_iterations - 1)

            should_print = (t % self.print_every) == 0

            if first_iteration or last_iteration or should_print:
                train_acc = self.accuracy(self.X_train, self.y_train, num_samples=self.num_train_samples)
                val_acc = self.accuracy(self.X_val, self.y_val,
                                        num_samples=self.num_val_samples) if self.X_val is not None else -1.0

                print('epoch:{}, iteration:{}, training accuracy: {}, validation accuracy: {}'.format(self.epoch, t,
                                                                                                      train_acc,
                                                                                                      val_acc))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
