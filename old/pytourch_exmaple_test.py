import torch
from torch.autograd import Function, Variable
from torch.nn import Module



class Stupied(Module):

    def __init__(self, param):
        super().__init__()
        self.param = param

    def forward(self, other):
        print(other)
        print(self.param)



Stupied("thing")("thing2")


