from torch import Tensor
from math import pi
import torch 
from botorch.test_functions.synthetic import SyntheticTestFunction

class Gramacy1d(SyntheticTestFunction):

    dim = 1
    _bounds = [(0.5, 2.5)]

    def evaluate_true(self, X:Tensor) -> Tensor:
        t1 = torch.sin(10*pi*X).div(2*X)
        t2 = X-1
        t2 = t2.pow(4)
        t3 = t1+t2
        return t3.squeeze()
    
class Gramacy2d(SyntheticTestFunction):

    dim = 2
    _bounds = [(-2, 6), (-2, 6)]

    def evaluate_true(self, X:Tensor) -> Tensor:
        t = X[:,0]* torch.exp(-X[:,0]**2 - X[:,1]**2)
        return t


class Higdon(SyntheticTestFunction):

    dim = 1
    _bounds = [(0, 20)]

    def evaluate_true(self, X:Tensor) -> Tensor:
        t1 = torch.sin(pi*X.div(5))
        t2 = 0.2*torch.cos(4*pi*X.div(5))
        t3 = t1+t2
        t4 = X.div(10) -1 
        t5 = torch.where(X <= 10,t3, t4)
        return t5.squeeze()
    
