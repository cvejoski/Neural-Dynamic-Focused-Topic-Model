from torch import nn


class MLP(nn.Module):

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.input_dim = kwargs.get("input_dim")
        self.output_dim = kwargs.get("output_dim")
        self.layers_dim = kwargs.get("layers_dim")
        self.dropout = kwargs.get("dropout", .4)

        self.layers_dim = [self.input_dim] + self.layers_dim + [self.output_dim]
        self.num_layer = len(self.layers_dim)
        self.output_transformation = kwargs.get("output_transformation")
        self.define_deep_parameters()

    def sample_parameters(self):
        parameters = {"input_dim": 2,
                      "layers_dim": [3, 2],
                      "ouput_dim": 3,
                      "normalization": True,
                      "ouput_transformation": None}
        return parameters

    def forward(self, x):
        return self.perceptron(x)

    def define_deep_parameters(self):
        self.perceptron = nn.ModuleList([])
        for layer_index in range(self.num_layer - 1):
            self.perceptron.append(nn.Linear(self.layers_dim[layer_index],
                                             self.layers_dim[layer_index + 1]))
            # self.perceptron.append(nn.BatchNorm1d(self.layers_dim[layer_index + 1]))
            if self.dropout > 0 and layer_index != self.num_layer-2:
                self.perceptron.append(nn.Dropout(self.dropout))
            if layer_index != self.num_layer-2:
                if layer_index < self.num_layer - 1 and self.num_layer > 2:
                    self.perceptron.append(nn.ReLU())
        if self.output_transformation == "relu":
            self.perceptron.append(nn.ReLU())
        elif self.output_transformation == "sigmoid":
            self.perceptron.append(nn.Sigmoid())
        self.perceptron = nn.Sequential(*self.perceptron)

    def init_parameters(self):
        for layer in self.perceptron:
            if hasattr(layer, 'weight'):
                if isinstance(layer, (nn.InstanceNorm2d, nn.LayerNorm)):
                    nn.init.normal_(layer.weight, mean=1., std=0.02)
                else:
                    nn.init.xavier_normal_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0.)

