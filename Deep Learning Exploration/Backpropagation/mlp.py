import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        W1, b1, W2, b2 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'], self.parameters['b2']

        if self.f_function == 'relu':
            f = torch.relu
        elif self.f_function == 'sigmoid':
            f = torch.sigmoid
        else:
            f = lambda x: x

        if self.g_function == 'relu':
            g = torch.relu
        elif self.g_function == 'sigmoid':
            g = torch.sigmoid
        else:
            g = lambda x: x

        self.cache['x'] = x
        self.cache['s1'] = torch.matmul(x, W1.T) + b1
        self.cache['a1'] = f(self.cache['s1'])
        self.cache['s2'] = torch.matmul(self.cache['a1'], W2.T) + b2
        self.cache['out'] = g(self.cache['s2'])
        return self.cache['out']
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        x, s1, a1, s2 = self.cache['x'], self.cache['s1'], self.cache['a1'], self.cache['s2']

        if self.f_function == 'relu':
            f_derivative = lambda x: (x > 0)
        elif self.f_function == 'sigmoid':
            f_derivative = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        else:
            f_derivative = lambda x: torch.ones_like(x)

        if self.g_function == 'relu':
            g_derivative = lambda x: (x > 0)
        elif self.g_function == 'sigmoid':
            g_derivative = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        else:
            g_derivative = lambda x: torch.ones_like(x)
        
        dJds2 = dJdy_hat * g_derivative(s2)                                     # (batch_size, linear_2_out_features)
        self.grads['dJdW2'] = torch.matmul(dJds2.T, a1) / dJdy_hat.shape[0]     # (linear_2_out_features, linear_2_in_features)
        self.grads['dJdb2'] = torch.mean(dJds2, axis=0)                         # (linear_2_out_features)
        
        dJda1 = torch.matmul(dJds2, self.parameters['W2'])                      # (batch_size, linear_1_out_features)
        dJds1 = dJda1 * f_derivative(s1)                                        # (batch_size, linear_1_out_features)
        self.grads['dJdW1'] = torch.matmul(dJds1.T, x) / dJdy_hat.shape[0]      # (linear_1_out_features, linear_1_in_features)
        self.grads['dJdb1'] = torch.mean(dJds1, axis=0)                         # (linear_1_out_features)
        return
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    error = y - y_hat
    squared_error = error ** 2
    J = torch.mean(squared_error)
    dJdy_hat = -2 * error / y.shape[1]
    return J, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    J = torch.mean(-(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)))
    dJdy_hat = (-(y / y_hat) + (1 - y) / (1 - y_hat)) / y.shape[1]
    return J, dJdy_hat