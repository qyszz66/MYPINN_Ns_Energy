import torch.nn as nn

# %%
class ConventBlock(nn.Module):
    def __init__(self, in_N, out_N):
        super(ConventBlock, self).__init__()
        self.Ls = None
        self.net = nn.Sequential(nn.Linear(in_N, out_N, bias=True), nn.Tanh())

    def forward(self, x):
        out = self.net(x)
        return out

class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        layers = []
        layers.append(ConventBlock(self.in_features, self.num_neurons))
        for i in range(0, self.num_layers-1):
            layers.append(ConventBlock(self.num_neurons, self.num_neurons))
         # output layer
        layers.append(nn.Linear(self.num_neurons, self.out_features, bias=True))
        # total layers
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        out = self.net(x)
        return out
