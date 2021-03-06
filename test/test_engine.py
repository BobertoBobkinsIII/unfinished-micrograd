"""Install pytest and run `pytest unit_tests.py`"""


import torch
from micrograd.engine import Value

def compare(func, *inputs):
    """Compare torch and micrograd implementations for func, applied to inputs.

    `func` takes in any number of arguments, each of which is a Value or size-1 Tesnor and returns
    any number of outputs.
    
    :param func: a mathematical function using Value or Tensor primitive functions. 
    :param inputs: a list of scalars, which are arguments to func.
    :returns: 
    :rtype:

    """
    assert len(inputs) > 0

    vs = [Value(x) for x in inputs]
    ts = [torch.Tensor([x]) for x in inputs]
    for t in ts:
        t.requires_grad = True
        
    # micrograd, pytorch output
    ov = func(*vs)
    ot = func(*ts)
    assert ov.data == ot.data.item(), f'values: {ov} != {ot}'

    ov.backward()
    ot.backward()
    for v, t in zip(vs, ts):
        assert v.grad == t.grad.item(), f'gradients: {v} != {t}'
  
  
def test_add():
    def add(x, y):
        return x + y
    compare(add, 5, 6)


def test_mul():
    pass


def test_pow():
    pass




  

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + (q * x)
    y.backward()
    xpt, ypt = x, y


    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


if __name__ == '__main__':
    test_sanity_check()
    test_more_ops()
