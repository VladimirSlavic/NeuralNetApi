# NeuralNetApi

Simple neural network API; it supports only fully connected layers. The api supports the cross entropy and MSE.
In terms of activations it supports ReLu and sigmoid. Data fed to neural network must be in the form
of numpy arrays. The following example shows how to construct a network with the architecture 40x50x50x5, trained with SGD
and relu activations

     small_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': None,
            'y_val': None,
        }

        learning_rate = 0.01

        model = NeuralNet(hidden_dims=[50, 50], input_dims=40,
                          num_classes=5,
                          loss_type=SoftMax(one_hot=True), function='relu', dtype=np.float128)

        self.solver = Controller(model, small_data,
                                 print_every=1000, num_epochs=20000, batch_size=50,
                                 update_rule='sgd',
                                 optim_config={
                                     'learning_rate': learning_rate,
                                 })
        self.solver.train()
