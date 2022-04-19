import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import normalize

from torch_geometric.nn import GCNConv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(1 * out_channels, 1 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Decoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * in_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(1 * in_channels, 1 * in_channels))
            self.conv.append(base_model(2 * in_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, in_channels)]
            for _ in range(1, k-1):
                self.conv = [base_model(in_channels, in_channels)]
            self.conv.append(base_model(in_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]

# class SelfExpression(nn.Module):
#     def __init__(self, n):
#         super(SelfExpression, self).__init__()
#         self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)
#         # self.Coefficient = Variable(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)
#         self.C_diag = torch.diag(torch.diag(self.Coefficient)).to(device)
#
#     def forward(self, x):  # shape=[n, d]
#         y = torch.matmul(self.Coefficient - self.C_diag, x)
#         return y, self.Coefficient - self.C_diag


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, num_sample: int):
        super(Model, self).__init__()
        self.n = num_sample
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(self.n, self.n, dtype=torch.float32), requires_grad=True)
        # self.self_expression = SelfExpression(self.n)

    def forward(self, x, edge_index):
        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        H = self.encoder(x, edge_index)
        C_diag = torch.diag(torch.diag(self.Coefficient)).to(device)
        # Coefficient = self.Coefficient - C_diag
        Coefficient = self.Coefficient
        CH = torch.matmul(Coefficient, H)
        X_ = self.decoder(CH, edge_index)

        # CH, Coefficient = self.self_expression(H)  # shape=[n, d]
        return H, CH, Coefficient, X_