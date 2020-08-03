

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

        out = Value(self.data+other.data,_children=(self,other),_op='+')

        def _backward():
            self.grad+=1
            other.grad+=1
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data*other.data,_children=(self,other),_op='*')

        def _backward():
            self.grad+=other.data
            other.grad+=self.data

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        out = Value(self.data**other,_children=(self,other),_op='**')

        def _backward():
            self.grad += other*self.data**(other-1)

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.max(np.array([0,self.data])),_children=(self),_op='ReLU')
        
        def _backward():
            if self.data <= 0:
                self.grad += 0
            if self.data > 0:
                self.grad += 1
        
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
        return self*-1

    def __radd__(self, other): # other + self
        return self+other

    def __sub__(self, other): # self - other
        return self+(-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self*other

    def __truediv__(self, other): # self / other
        return self*other**-1

    def __rtruediv__(self, other): # other / self
        return other*self**-1
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
