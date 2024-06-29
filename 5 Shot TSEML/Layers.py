import torch, math
import torch.nn.functional as F
import torch.nn as nn

class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super(LinearLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input, parameter_name_to_parameter=None):

        if not parameter_name_to_parameter is None:
            weight = parameter_name_to_parameter['weight']
            bias = parameter_name_to_parameter['bias']

        else:
            weight = self.weight
            bias = self.bias

        return F.linear(input, weight, bias)

class Conv2dLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):

        super(Conv2dLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, parameter_name_to_parameter=None):

        if not parameter_name_to_parameter is None:
            weight = parameter_name_to_parameter['weight']
            bias = parameter_name_to_parameter['bias']

        else:
            weight = self.weight
            bias = self.bias

        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        groups = self.groups

        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

class MultipliedLayer(nn.Module):

    def __init__(self, shape):

        super(MultipliedLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(*shape))

        nn.init.zeros_(self.weight)

    def forward(self, input, parameter_name_to_parameter=None):

        if not parameter_name_to_parameter is None:
            weight = parameter_name_to_parameter['weight']

        else:
            weight = self.weight

        return torch.multiply(input, torch.sigmoid(weight))

class BatchNorm2dLayer(nn.Module):

    def __init__(self, num_features):

        super(BatchNorm2dLayer, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, parameter_name_to_parameter=None):

        if not parameter_name_to_parameter is None:
            weight = parameter_name_to_parameter['weight']
            bias = parameter_name_to_parameter['bias']

        else:
            weight = self.weight
            bias = self.bias

        return F.batch_norm(input, None, None, weight, bias, True)
    
class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.parameter_name_list = list()

    def forward(self, input, parameter_list=None):

        input = torch.reshape(input, self.input_shape)

        if not parameter_list is None:
            parameter_name_list = self.parameter_name_list

            index_to_parameter_name_to_parameter = dict()

            for name, parameter in zip(parameter_name_list, parameter_list):

                (index, parameter_name) = name.split('.')
                index = int(index)

                if not index_to_parameter_name_to_parameter.__contains__(index):
                    parameter_name_to_parameter = dict()

                else:
                    parameter_name_to_parameter = index_to_parameter_name_to_parameter[index]

                parameter_name_to_parameter[parameter_name] = parameter
                index_to_parameter_name_to_parameter[index] = parameter_name_to_parameter

            for index in range(len(self.net)):

                layer = self.net[index]

                if index_to_parameter_name_to_parameter.__contains__(index):
                    if isinstance(layer, LinearLayer) or isinstance(layer, Conv2dLayer) or isinstance(layer, MultipliedLayer) or isinstance(layer, BatchNorm2dLayer):
                        input = layer(input, index_to_parameter_name_to_parameter[index])

                else:
                    input = layer(input)

        else:
            input = self.net(input)

        return input

class EmbeddingModel(Model):

    def __init__(self):

        super(EmbeddingModel, self).__init__()

        self.input_shape = [-1, 1, 130, 158]

        self.net = nn.Sequential(
            Conv2dLayer(1, 32, [130, 1]),
            nn.ReLU(),
            nn.MaxPool2d([1, 2]),
            nn.Flatten(),
            LinearLayer(2528, 112),
            nn.ReLU()
        )

        for param_name, param in self.net.named_parameters():
            self.parameter_name_list.append(param_name)

class ClassificationLayer(Model):

    def __init__(self, way):

        super(ClassificationLayer, self).__init__()

        self.input_shape = [-1, 112]

        self.net = nn.Sequential(
            LinearLayer(112, way)
        )

        for param_name, param in self.net.named_parameters():
            self.parameter_name_list.append(param_name)

class ClassificationModel(Model):

    def __init__(self, way):

        super(ClassificationModel, self).__init__()

        self.input_shape = [-1, 1, 130, 158]

        self.net = nn.Sequential(
            Conv2dLayer(1, 32, [130, 1]),
            nn.ReLU(),
            nn.MaxPool2d([1, 2]),
            nn.Flatten(),
            LinearLayer(2528, 112),
            nn.ReLU(),
            LinearLayer(112, way)
        )

        for param_name, param in self.net.named_parameters():
            self.parameter_name_list.append(param_name)

class EmbeddingModelWithoutLastReLU(Model):

    def __init__(self):

        super(EmbeddingModelWithoutLastReLU, self).__init__()

        self.input_shape = [-1, 1, 130, 158]

        self.net = nn.Sequential(
            Conv2dLayer(1, 32, [130, 1]),
            nn.ReLU(),
            nn.MaxPool2d([1, 2]),
            nn.Flatten(),
            LinearLayer(2528, 112)
        )

        for param_name, param in self.net.named_parameters():
            self.parameter_name_list.append(param_name)