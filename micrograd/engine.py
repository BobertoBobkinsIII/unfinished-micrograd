

class Value:
    """Stores a scalar or vector value and its gradient.

    Performing operations on values builds up a computation graph where each value is a node and
    self._prev contains the nodes pointing to this node. This is used by the backpropagation
    algorithm.

    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        """Allows two values to be added together with '+'.

        Sets the _backward method for the resulting Value object, which is called during backprop.

        :returns: the resulting Value object.

        """
        other = other if isinstance(other, Value) else Value(other)

        out = None
        raise NotImplementedError('TODO: __add__')

        def _backward():
            raise NotImplementedError('TODO: backward for __add__')
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = None
        raise NotImplementedError('TODO: __mul__')

        def _backward():
            raise NotImplementedError('TODO: backward for __mul__')

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        out = None
        raise NotImplementedError('TODO: __pow__')

        def _backward():
            raise NotImplementedError('TODO: backward for __pow__')

        out._backward = _backward
        return out

    def relu(self):
        out = None
        raise NotImplementedError('TODO: relu')

        def _backward():
            raise NotImplementedError('TODO: backward for relu')
        
        out._backward = _backward
        return out

    def backward(self):
        """Perform backpropagation on this Value.

        For ever Value object v in self's computation graph, set v.grad to the partial derivative
        of self with respect to v.

        Recursively calls v._backward() for all such v.

        """

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        raise NotImplementedError('neg')

    def __radd__(self, other): # other + self
        raise NotImplementedError('radd')

    def __sub__(self, other): # self - other
        raise NotImplementedError('sub')

    def __rsub__(self, other): # other - self
        raise NotImplementedError('rsub')

    def __rmul__(self, other): # other * self
        raise NotImplementedError('rmul')

    def __truediv__(self, other): # self / other
        raise NotImplementedError('truediv')

    def __rtruediv__(self, other): # other / self
        raise NotImplementedError('rtruediv')

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
