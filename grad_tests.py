import math
from autograd import Value
from nn import MLP
import torch
from autograd import Value


### TEST AGAINST PYTORCH
def test_scalar_vs_torch():
    
    x = Value(2.0)
    y = Value(3.0)
    z = (x * y + x) ** 2
    z.backward()


    xt = torch.tensor(2.0, requires_grad=True)
    yt = torch.tensor(3.0, requires_grad=True)
    zt = (xt * yt + xt) ** 2
    zt.backward()


    assert abs(z.data - zt.item()) < 1e-6
    assert abs(x.grad - xt.grad.item()) < 1e-6
    assert abs(y.grad - yt.grad.item()) < 1e-6



def test_add_mul():
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x
    z.backward()

    # z = xy + x
    # dz/dx = y + 1 = 4
    # dz/dy = x = 2
    assert abs(x.grad - 4.0) < 1e-6
    assert abs(y.grad - 2.0) < 1e-6

def test_power_scalar():
    x = Value(3.0)
    z = x ** 2
    z.backward()

    # z = x^2 → dz/dx = 2x = 6
    assert abs(z.data - 9.0) < 1e-6
    assert abs(x.grad - 6.0) < 1e-6

def test_power_value():
    x = Value(3.0)
    y = Value(2.0)
    z = x ** y
    z.backward()

    # dz/dx = y * x^(y-1) = 2 * 3 = 6
    # dz/dy = x^y * ln(x) = 9 * ln(3)
    assert abs(x.grad - 6.0) < 1e-6
    assert abs(y.grad - (9 * math.log(3))) < 1e-6

def test_relu():
    x = Value(-2.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0

    x = Value(3.0)
    y = x.relu()
    y.backward()
    assert y.data == 3.0
    assert x.grad == 1.0

def test_sigmoid():
    x = Value(0.0)
    y = x.sigmoid()
    y.backward()

    # sigmoid(0) = 0.5
    # derivative = s(1-s) = 0.25
    assert abs(y.data - 0.5) < 1e-6
    assert abs(x.grad - 0.25) < 1e-6

def test_mlp_backward():
    mlp = MLP(2, [3, 1])

    x = [Value(1.0), Value(-1.0)]
    y_pred = mlp(x)[0]

    y_pred.backward()

    # At least one parameter must receive gradient
    grads = [p.grad for p in mlp.parameters()]
    assert any(abs(g) > 0 for g in grads)


test_add_mul()
test_power_scalar()
test_power_value()
test_relu()
test_sigmoid()
test_mlp_backward()
print('passed')