import torch
import torch.nn as nn
import torch.nn.functional as F


## Define a activation function that simulate IF neuron
class Model(torch.nn.Module):

    def __init__(self, upper_bound=50):
        """
        layer initialization
        """
        super(Model, self).__init__()
        self.upper_bound = upper_bound
        self.if_activate = if_activate()
        self.conv1 = torch.nn.Conv2d(6, 12, kernel_size=5, stride=2, padding=1)
        self.conv1_bn = torch.nn.BatchNorm2d(12)
        self.conv2 = torch.nn.Conv2d(12, 24, kernel_size=5, stride=2, padding=1)
        self.conv2_bn = torch.nn.BatchNorm2d(24)
        self.conv3 = torch.nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=1)
        self.conv3_bn = torch.nn.BatchNorm2d(48)
        self.conv4 = torch.nn.Conv2d(48, 96, kernel_size=5, stride=2, padding=1)
        self.conv4_bn = torch.nn.BatchNorm2d(96)
        self.linear = torch.nn.Linear(192, 360)

    def forward(self, x):
        """
        define the CNN model
        """
        x1 = self.if_activate.apply(self.conv1_bn(self.conv1(x)), self.upper_bound)
        x2 = self.if_activate.apply(self.conv2_bn(self.conv2(x1)), self.upper_bound)
        x3 = self.if_activate.apply(self.conv3_bn(self.conv3(x2)), self.upper_bound)
        x4 = self.if_activate.apply(self.conv4_bn(self.conv4(x3)), self.upper_bound)
        x4 = x4.view(-1, 192)
        y_pred = torch.sigmoid(self.linear(x4))

        return x1, x2, x3, x4, y_pred


class if_activate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, upper_bound):
        ctx.save_for_backward(input)
        output = torch.floor(input)
        output = torch.clamp(output, min=0, max=upper_bound)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, = ctx.saved_tensors
        grad_input = grad_out.clone()
        grad_input[input < 0] = 0
        return grad_input, None
