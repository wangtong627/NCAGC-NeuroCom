import argparse
import os.path as osp
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import contrastive_loss
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import to_dense_adj
from dataset import get_dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array, check_symmetric
import scipy.sparse as sparse
import clustering_metric

from simple_param.sp import SimpleParam
from model import Encoder, Decoder, Model
from sklearn.neighbors import NearestNeighbors


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 1,
            heads=1
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]

def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]

# def thrC(C, alpha):
# #     if alpha < 1:
# #         N = C.shape[1]
# #         Cp = np.zeros((N, N))
# #         S = np.abs(np.sort(-np.abs(C), axis=0))
# #         Ind = np.argsort(-np.abs(C), axis=0)
# #         for i in range(N):
# #             cL1 = np.sum(S[:, i]).astype(float)
# #             stop = False
# #             csum = 0
# #             t = 0
# #             while (stop == False):
# #                 csum = csum + S[t, i]
# #                 if csum > alpha * cL1:
# #                     stop = True
# #                     Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
# #                 t = t + 1
# #     else:
# #         Cp = C
# #     return Cp
# #
# #
# # def post_proC(C, K, d, ro):
# #     # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
# #     n = C.shape[0]
# #     C = 0.5 * (C + C.T)
# #     # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
# #     r = d * K + 1
# #     U, S, _ = svds(C, r, v0=np.ones(n))
# #     U = U[:, ::-1]
# #     S = np.sqrt(S[::-1])
# #     S = np.diag(S)
# #     U = U.dot(S)
# #     U = normalize(U, norm='l2', axis=1)
# #     Z = U.dot(U.T)
# #     Z = Z * (Z > 0)
# #     L = np.abs(Z ** ro)
# #     L = L / L.max()
# #     L = 0.5 * (L + L.T)
# #     spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
# #                                           assign_labels='discretize')
# #     spectral.fit(L)
# #     grp = spectral.fit_predict(L)
# #     return grp, L

# def spectral_clustering_1(C, K, d, alpha, ro):
#     C = thrC(C, alpha)
#     y, _ = post_proC(C, K, d, ro)
#     return y

def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while stop == False:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


# def spectral_clustering_2(affinity_matrix_, n_clusters, k, seed=1, n_init=20):
#     affinity_matrix_ = check_symmetric(affinity_matrix_)
#     random_state = check_random_state(seed)
#
#     laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
#     _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian,
#                                  k=k, sigma=None, which='LA')
#     embedding = normalize(vec)
#     _, labels_, _ = cluster.k_means(embedding, n_clusters,
#                                          random_state=seed, n_init=n_init)
#     return labels_


def clustering_evaluation(predict_labels, true_labels):
    acc, nmi, ari = clustering_metric.evaluationClusterModelFromLabel(true_labels, predict_labels)
    return acc, nmi, ari


def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    f = lambda x: torch.exp(x /param['tau'])
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def instanceloss(z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
    l1 = semi_loss(z1, z2)
    l2 = semi_loss(z2, z1)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret


def knbrsloss(H, k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(H.cpu().detach().numpy())
    _, indices = nbrs.kneighbors(H.cpu().detach().numpy())
    f = lambda x: torch.exp(x / param['tau_knbrs'])
    refl_sim = f(sim(H, H))
    V = torch.zeros((param['instance_number'], k)).to(device)
    for i in range(param['instance_number']):
        for j in range(k):
            V[i][j] += refl_sim[i][indices[i][j + 1]]
    ret = -torch.log(
        V.sum(1) / (refl_sim.sum(1) - refl_sim.diag()))
    ret = ret.mean()
    return ret


def train():
    model.train()
    optimizer.zero_grad()
    H, CH, Coefficient, X_ = model(data.x, data.edge_index)
    loss_knbrs = knbrsloss(H, k = 10)
    # C_diag = torch.diag(torch.diag(Coefficient))
    # A_ = torch.sigmoid(torch.mm(CH, CH.t()))
    # pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
    # norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
    # loss_edge = norm * F.binary_cross_entropy_with_logits(A_, A, pos_weight=pos_weight)
    # rec_loss = torch.sum(torch.pow(H - CH, 2))
    # loss_instance = criterion_instance(H, CH)
    rec_loss = torch.sum(torch.pow(data.x - X_, 2))
    # rec_loss = instanceloss(data.x, X_)
    loss_instance = instanceloss(H, CH)
    loss_coef = torch.sum(torch.pow(Coefficient, 2))
    # loss_coef = torch.mean(torch.pow(Coefficient, 2))
    # loss_coef = regularizer(Coefficient)
    loss = 1.0 * loss_instance + 1.0 * loss_knbrs + 1.0 * loss_coef + 0.1 * rec_loss
    loss.backward()
    optimizer.step()
    return loss_instance.item(), loss_coef.item(), loss_knbrs.item(), rec_loss.item(), loss.item(), Coefficient


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--param', type=str, default='local:Cora.json')
    parser.add_argument('--seed', type=int, default=39788)
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'num_epochs': 3000,
        'weight_decay': 1e-5,
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]
    # print(min(data.y))
    data = data.to(device)
    # A = to_dense_adj(data.edge_index)
    # A = A.view([param['instance_number'], param['instance_number']])

    criterion_instance = contrastive_loss.InstanceLoss(param['instance_number'], param['tau'], device).to(
        device)

    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    decoder = Decoder(param['num_hidden'], dataset.num_features, get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = Model(encoder, decoder, param['instance_number']).to(device)
    # H, CH, Coefficient = model(data.x, data.edge_index)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    K = len(np.unique(data.y.cpu().numpy()))
    # dim_subspace = 12
    # alpha = 0.04
    alpha = max(0.4 - (K - 1) / 10 * 0.1, 0.1)
    # ro = 8
    # num_subspaces = K
    # spectral_dim = K

    acclist = []
    nmilist = []
    arilist = []

    for epoch in range(1, param['num_epochs']+1): #param['num_epochs'] + 1
        loss_instance, loss_c, loss_knbrs, rec_loss, loss, C = train()

        # get C
        C = C.detach().to('cpu').numpy()

        commonZ = thrC(C, alpha)
        y_pred, _ = post_proC(commonZ, K)
        # y_pred = spectral_clustering_1(C, K, dim_subspace, alpha, ro)

        # Evalue
        # rows = list(range(param['instance_number']))
        # cols = list(range(param['instance_number']))
        # C[rows, cols] = 0.0
        # C_normalized = normalize(C).astype(np.float32)
        # Aff = 0.5 * (np.abs(C) + np.abs(C).T) # get Aff
        # y_pred = spectral_clustering_2(Aff, num_subspaces, spectral_dim) # use spectral_clustering get predicted label
        acc, nmi, ari = clustering_evaluation(y_pred, data.y.cpu().numpy()) # get acc nmi ari

        acclist.append(acc)
        nmilist.append(nmi)
        arilist.append(ari)

        print(f'Epoch={epoch:03d}, loss={loss:.4f}, loss_instance = {loss_instance:.4f}, loss_knbrs = {loss_knbrs:.4f}, rec_loss = {rec_loss:.4f}, loss_c={loss_c:.4f}')
        print('ACC = {:.4f}, NMI = {:.4f} ARI = {:.4f} '.format(acc, nmi, ari))
        # print(C)

    print("=== Final ===")
    print('best_acc: {}, best_nmi: {}, best_adj: {}'.format(max(acclist), max(nmilist), max(arilist)))

    x = np.arange(param['num_epochs'])
    plt.plot(x, acclist, label='accdsc')
    plt.plot(x, nmilist, label='nmidsc')
    plt.plot(x, arilist, label='aridsc')
    # plt.plot(x, list_loss, label='list_loss')
    plt.legend(['ACC', 'NMI', 'ARI'])
    plt.title("Cora_Performance")
    plt.show()