import random
from micrograd.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        """Perform the neuron's forward pass computation (inference). 

        :param x: a list of Value objects, i.e. a vector.
        :returns: a Value object containing relu(w^T x + b) if self.nonlin, else w^T x + b.

        """
        
        raise NotImplementedError('TODO: Neuron.__call__')

    def parameters(self):
        """Return the list of Value objects which make up the weights for this neuron.

        :returns: list of weight Values, with w terms then the bias term last.

        """
        raise NotImplementedError('TODO: Neuron.parameters')

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

    
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """FIXME! briefly describe function

        :param x: a list of Value objects, i.e. a vector.
        :returns: a list of Value objects, if nin > 1, else a single Value object.

        """
        raise NotImplementedError('Layer.__call__')

    def parameters(self):
        """Get the parameters for this neuron.

        If the lsit of neurons is ns, then the order should be 
        [ns[0].parameters()[0], ns[0].parameters()[1], ..., ns[0].parameters()[-1],
         ns[1].parameters()[0], ns[1].parameters()[1], ..., ns[1].parameters()[-1],
         ...,
         ns[-1].parameters()[0], ns[-1].parameters()[1], ..., ns[-1].parameters()[-1]
        ]
        
        But DO NOT call parameters() this many times. This function can be a one-liner.

        :returns: list of Value objects for the parameters in self.

        """
        
        raise NotImplementedError('Layer.parameters')

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    
class MLP(Module):
    def __init__(self, nin, nouts):
        """Construct the multi-layer perceptron (fully-connected network).

        :param nin: input dimension of the network.
        :param nouts: list of layer output sizes for each layer in the network.

        """
        raise NotImplementedError('MLP.__init__')

    def __call__(self, x):
        """Forward pass for this neural network."""
        raise NotImplementedError('MLP.__call__')

    def parameters(self):
        """Get the parameters of the neural network in the same order as the layers."""
        raise NotImplementedError('MLP.parameters')

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
