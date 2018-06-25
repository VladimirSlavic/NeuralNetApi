from builtins import range

import numpy as np

from loss_capabilities.softmax_loss import SoftMax


class NeuralNet:
    def __init__(self, hidden_dims=[50, 50], input_dims=40, num_classes=5, reg=0.0, weight_dev=1e-2, dtype=np.float32,
                 loss_type=None, function='sigmoid'):  # function='sigmoid'

        self.num_layers = len(hidden_dims) + 1
        self.dtype = dtype
        self.params = {}
        self.reg = reg
        self.function = function

        if loss_type is None:
            self.loss_type = SoftMax()
        else:
            self.loss_type = loss_type

        true_net_structure = [input_dims] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            w_name = 'W' + str(i + 1)
            b_name = 'b' + str(i + 1)

            # scale = weight_dev
            w_scale = pow(true_net_structure[i], -0.5)
            self.params[b_name] = np.zeros(shape=(true_net_structure[i + 1]))
            self.params[w_name] = np.random.normal(scale=w_scale,
                                                   size=(true_net_structure[i], true_net_structure[i + 1]))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def forward_pass(self, x, w, b):

        # reshape based on shape type, dakle NxM ostavit, ali ako NxM1xM2xM3 onda pretvorit sa prod
        X_reshaped = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        out = X_reshaped.dot(w) + b
        cache = (x, w, b)
        return out, cache  # per layer pass

    def sigmoid_forward(self, x):
        a = 1 / (1.0 + np.exp(-x))
        return a, x

    def relu_forward(self, x):
        out = None
        out = np.maximum(x, 0)
        cache = x
        return out, cache

    def relu_backward(self, dout, cache):
        dx, x = None, cache
        x[x > 0] = 1
        x[x < 0] = 0
        dx = x * dout
        return dx

    def forwawrd_pass_activation(self, x, w, b, activation_function='sigmoid'):
        a, fc_cache = self.forward_pass(x, w, b)
        if activation_function == 'sigmoid':
            out, activation_cache = self.sigmoid_forward(a)  # activation cache je samo a=x*w + b
        else:
            out, activation_cache = self.relu_forward(a)
        return out, (fc_cache, activation_cache)

    def _backward_sigmoid(self, dout, cache):
        """
        Backward pass of upstream gradient through an activation, currently only sigmoid
        :param dout: upstream gradient
        :param cache: hold the vale of activation 'a' after a sigmoid/relu/leaky_relu itd
        :return: gradijent od x
        """
        dx, x = None, cache
        sub = lambda param: 1.0 / (1.0 + np.exp(-param))

        x = sub(x) * (1 - sub(x))
        dx = x * dout

        return dx

    def backward_pass(self, dout, cache):
        """

        :param dout: upstream gradient
        :param cache: (x, w, b) from previous layer
        :return:
        """
        x, w, b = cache
        dx, dw, db = None, None, None

        x_reshaped = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        dx = dout.dot(w.T)
        dw = x_reshaped.T.dot(dout)
        db = np.sum(dout, axis=0)

        return dx, dw, db

    def back_pass_activation(self, dout, cache, activation_function='sigmoid'):
        fc_cache, activation_cache = cache
        if activation_function == 'relu':
            da = self.relu_backward(dout, activation_cache)
        elif activation_function == 'sigmoid':
            da = self._backward_sigmoid(dout=dout, cache=activation_cache)
        dx, dw, db = self.backward_pass(dout=da, cache=fc_cache)
        return dx, dw, db

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        self.cache = {}
        score = X

        for i in range(1, self.num_layers + 1):
            id = str(i)
            w_name = 'W' + id
            b_name = 'b' + id
            cache_name = 'c' + id

            if i == self.num_layers and self.loss_type.get_name() == 'softmax':
                score, self.cache[cache_name] = self.forward_pass(score, self.params[w_name], self.params[b_name])
            elif i == self.num_layers and self.loss_type.get_name() == 'quadratic':
                score, self.cache[cache_name] = self.forwawrd_pass_activation(score, self.params[w_name],
                                                                              self.params[b_name],
                                                                              activation_function='sigmoid')
            else:
                score, self.cache[cache_name] = self.forwawrd_pass_activation(score, self.params[w_name],
                                                                              self.params[b_name], self.function)
                # nema tu vise aktivacija

        if mode == 'test':
            return score

        loss, grads = 0.0, {}

        loss, upstream_gradient = self.loss_type.loss(score, y)  ##
        # print('current loss is: {}'.format(loss))

        for i in range(self.num_layers, 0, -1):
            id = str(i)
            w_name = 'W' + id
            b_name = 'b' + id
            cache_name = 'c' + id

            loss += 0.5 * self.reg * np.sum(self.params[w_name] ** 2)  # regularizacija

            if i == self.num_layers and self.loss_type.get_name() == 'softmax':
                upstream_gradient, grads[w_name], grads[b_name] = self.backward_pass(upstream_gradient,
                                                                                     self.cache[cache_name])
            elif i == self.num_layers and self.loss_type.get_name() == 'quadratic':
                upstream_gradient, grads[w_name], grads[b_name] = self.back_pass_activation(upstream_gradient,
                                                                                            self.cache[cache_name],
                                                                                            activation_function='sigmoid')
            else:
                upstream_gradient, grads[w_name], grads[b_name] = self.back_pass_activation(upstream_gradient,
                                                                                            self.cache[cache_name],
                                                                                            self.function)

            grads[w_name] += self.reg * self.params[w_name]

        return loss, grads
