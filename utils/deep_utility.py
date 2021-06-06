import os

import mdtraj as md
import numpy as np
import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
# from torch_geometric.utils import subgraph
from torch_sparse import coalesce

from layers import *
from utils.dataset import AutoMDS, DeepMDDataLoader, DeepMDDataset

EPS = 1e-15
MAX_LOGSTD = 10


def VAMP_loss(X0_t, X1_t, lag: int):
    def _inv(x, ret_sqrt=False):
        epsilon = 1e-10

        # Calculate eigvalues and eigvectors
        eigval, eigvec = torch.linalg.eigh(x)

        index_eig = torch.masked_select(torch.arange(eigval.shape[0]), eigval > epsilon)
        eigval = eigval[index_eig]
        eigvec = (eigvec.t()[index_eig]).t()

        # Build the diagonal matrix with the filtered eigenvalues or square
        # root of the filtered eigenvalues according to the parameter
        diag = torch.diag(torch.sqrt(1 / eigval)) if ret_sqrt else torch.diag(1 / eigval)

        # Rebuild the square root of the inverse matrix x @ x_inv = 1
        x_inv = eigvec.t() @ (diag @ eigvec)
        return x_inv

    batch = X0_t.shape[0]

    # shapes: n_batch, n_dimensions
    X0_t_mf = (X0_t[:-lag] - torch.mean(X0_t[:-lag], dim=0)).t()
    X1_t_mf = (X1_t[:-lag] - torch.mean(X1_t[:-lag], dim=0)).t()
    X1_tt_mf = (X1_t[lag:] - torch.mean(X1_t[lag:], dim=0)).t()

    # shapes: (n_dimensions, n_batch) @ (n_batch, n_dimensions) -> (n_dimensions, n_dimensions)
    C_00 = X0_t_mf @ X0_t_mf.t() / batch
    C_11 = X1_t_mf @ X1_t_mf.t() / batch
    C_01 = X0_t_mf @ X1_tt_mf.t() / batch

    cov_00_inv = _inv(C_00, ret_sqrt=True)
    cov_11_inv = _inv(C_11, ret_sqrt=True)

    vamp_matrix = torch.matmul(torch.matmul(cov_00_inv, C_01), cov_11_inv)
    vamp_score = torch.norm(vamp_matrix)

    return - torch.square(vamp_score)


def loss_VAMP2_autograd(y_pred):
    def _inv(x, ret_sqrt=False):
        epsilon = 1e-10

        # Calculate eigvalues and eigvectors
        eigval, eigvec = torch.linalg.eigh(x)

        # Build the diagonal matrix with the filtered eigenvalues or square
        # root of the filtered eigenvalues according to the parameter
        diag = torch.diag(torch.sqrt(1 / eigval)) if ret_sqrt else torch.diag(1 / eigval)

        # Rebuild the square root of the inverse matrix x @ x_inv = 1
        x_inv = eigvec.t() @ (diag @ eigvec)
        return x_inv

    def _prep_data(data: Tensor):
        shape = data.shape
        b = shape[0]
        o = shape[1] // 2

        # Split the data of the two networks and transpose it
        x_biased = data[:, :o].t()
        y_biased = data[:, o:].t()

        # Subtract the mean
        x = x_biased - torch.mean(x_biased, dim=1, keepdim=True)
        y = y_biased - torch.mean(y_biased, dim=1, keepdim=True)

        return x, y, b, o

    # Remove the mean from the data
    x, y, batch_size, output_size = _prep_data(y_pred)

    # Calculate the covariance matrices
    cov_01 = 1 / (batch_size - 1) * torch.matmul(x, y.t())
    cov_00 = 1 / (batch_size - 1) * torch.matmul(x, x.t())
    cov_11 = 1 / (batch_size - 1) * torch.matmul(y, y.t())

    # Calculate the inverse of the self-covariance matrices
    cov_00_inv = _inv(cov_00, ret_sqrt=True)
    cov_11_inv = _inv(cov_11, ret_sqrt=True)

    vamp_matrix = torch.matmul(torch.matmul(cov_00_inv, cov_01), cov_11_inv)
    vamp_score = torch.norm(vamp_matrix)

    return - torch.square(vamp_score)


def trj_batch_to_list(trj_batch, batch: Tensor):
    trj = torch.split(trj_batch.transpose(0, 1), [torch.sum((batch == i)) for i in torch.arange(0, batch.max() + 1)])
    trj = [t.transpose(0, 1).view(t.shape[1], -1) for t in trj]
    # Shape trj_batch: n_frames, batch, 1
    # Shape trj:       n_batches, n_frames, n_edges, 1
    # batch [0, 0, 1, 1, 1, 1, ... , n_batches, n_batches, n_batches]
    return trj

def test(data_loader: DeepMDDataLoader, model, test_iter=20):
    losses = []
    with torch.no_grad():
        data_loader.test()
        for epoch, (batch, trj) in enumerate(data_loader):
            if epoch > 10:
                break
            trj_out = model(batch, test_iter)
            loss = model.energy_loss(trj, trj_out)
            losses.append(loss.item())
        data_loader.train()
    print(f'average test loss: {np.array(losses).mean():>5.3f} +- {np.array(losses).std():>5.3f}')
    return np.array(losses).mean()

def mask(feat_tensor, feat_column, value=1.0):
    idx = torch.masked_select(torch.arange(feat_tensor.shape[0]), feat_tensor[:, feat_column] == value)
    return feat_tensor[idx], idx


def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False, num_nodes=None):
    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1

        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.size(0), device=device)

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr, mask


def get_subgraph(x, edge_index, edge_attr, feat_column, value=1.0, renumber=False) -> Tuple[Tensor, torch.LongTensor, Tensor, Tensor, torch.LongTensor]:
    idx_mask = x[:, feat_column] == value
    sub_edge_index, sub_edge_attr, mask = subgraph(idx_mask, edge_index, edge_attr, renumber, x.shape[0])
    sub_x = x[torch.masked_select(torch.arange(x.shape[0]), idx_mask)]
    return sub_x, sub_edge_index, sub_edge_attr, mask, idx_mask


def get_subgraph_mask(x, edge_index, edge_attr, mask, renumber=False) -> Tuple[Tensor, torch.LongTensor, Tensor, Tensor]:
    sub_edge_index, sub_edge_attr, mask_edges = subgraph(mask, edge_index, edge_attr, renumber, x.shape[0])
    return x, sub_edge_index, sub_edge_attr, mask_edges


def compute_covariance(x: Tensor):
    # Shape: [*, n_feat]
    N = x.shape[0]
    x_meanf = (x - torch.mean(x, dim=0)).t()
    C = x_meanf @ x_meanf.t() / (N - 1)
    return C


def reduce_edges_length(edges_length: torch.Tensor):
    idx1 = torch.arange(0, edges_length.shape[0], 2, device=edges_length.device, dtype=torch.long)
    idx2 = idx1 + torch.ones_like(idx1, device=edges_length.device, dtype=torch.long)
    return (edges_length[idx1] + edges_length[idx2]) / 2.0


def augment_edges_idx(edges_idx: torch.Tensor):
    r = torch.zeros((2, edges_idx.shape[1] * 2), dtype=torch.long, device=edges_idx.device)
    row, col = edges_idx
    idx1 = torch.arange(0, edges_idx.shape[1] * 2, 2, device=edges_idx.device, dtype=torch.long)
    idx2 = idx1 + 1
    r[0, idx1] = row
    r[1, idx1] = col
    r[0, idx2] = col
    r[1, idx2] = row
    return r


def augment_edges_length(edges_length: torch.Tensor):
    if len(edges_length.shape) == 1:
        r = torch.zeros((edges_length.shape[0] * 2), dtype=edges_length.dtype, device=edges_length.device)
    else:
        r = torch.zeros((edges_length.shape[0] * 2, edges_length.shape[1]), dtype=edges_length.dtype, device=edges_length.device)

    idx1 = torch.arange(0, edges_length.shape[0] * 2, 2, device=edges_length.device, dtype=torch.long)
    idx2 = idx1 + 1
    r[idx1] = edges_length
    r[idx2] = edges_length
    return r


def symmetric_edges_length(edges_length: torch.Tensor):
    idx1 = torch.arange(0, edges_length.shape[0], 2, device=edges_length.device, dtype=torch.long)
    idx2 = idx1 + torch.ones_like(idx1, device=edges_length.device, dtype=torch.long)
    out_edges_length = edges_length.clone()
    out_edges_length[idx1] = (edges_length[idx1] + edges_length[idx2]) / 2.0
    out_edges_length[idx2] = (edges_length[idx1] + edges_length[idx2]) / 2.0
    return out_edges_length


def get_line_graph(edge_index, num_nodes, force_directed=False):
    N = num_nodes
    (row, col), _ = coalesce(edge_index, None, N, N)

    if force_directed:
        i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

        count = scatter_add(torch.ones_like(row), row, dim=0,
                            dim_size=num_nodes)
        cumsum = torch.cat([count.new_zeros(1), count.cumsum(0)], dim=0)

        cols = [
            i[cumsum[col[j]]:cumsum[col[j] + 1]]
            for j in range(col.size(0))
        ]
        rows = [row.new_full((c.numel(),), j) for j, c in enumerate(cols)]

        row, col = torch.cat(rows, dim=0), torch.cat(cols, dim=0)

        edge_index = torch.stack([row, col], dim=0)

    else:
        # Compute node indices.
        mask = row < col
        row, col = row[mask], col[mask]
        i = torch.arange(row.size(0), dtype=torch.long, device=row.device)

        (row, col), i = coalesce(
            torch.stack([
                torch.cat([row, col], dim=0),
                torch.cat([col, row], dim=0)
            ], dim=0), torch.cat([i, i], dim=0), N, N)

        # Compute new edge indices according to `i`.
        count = scatter_add(torch.ones_like(row), row, dim=0,
                            dim_size=num_nodes)
        joints = torch.split(i, count.tolist())

        def generate_grid(x):
            row = x.view(-1, 1).repeat(1, x.numel()).view(-1)
            col = x.repeat(x.numel())
            return torch.stack([row, col], dim=0)

        joints = [generate_grid(joint) for joint in joints]
        joints = torch.cat(joints, dim=1)
        joints, _ = remove_self_loops(joints)
        N = row.size(0) // 2
        joints, _ = coalesce(joints, None, N, N)

        edge_index = joints

    return edge_index


def reconstruct_traj(trj_out: Tensor, pdb_name, data_loader: DeepMDDataLoader):
    dataset = data_loader.dataset
    trj = md.load_pdb(os.path.join(dataset.root, pdb_name, pdb_name + '_noh.pdb'))

    batch, trj_p = dataset.get_trajectory(pdb_name=pdb_name, file_id=0, frame_i=0, frame_j=-1, lag=1)

    mds = AutoMDS(trj.n_atoms, 5000, 1e-4)
    mds = mds.to(data_loader.device)
    mds.fit_edges(batch.edge_index, trj_p[0].detach())
    # trj.xyz = np.array([])
    trj.xyz = np.append(trj.xyz, [mds.pos.detach().cpu().numpy()], axis=0)
    trj.save(os.path.join('plots', f'{pdb_name}.pdb'))
    for i in range(trj_out.shape[0]):
        mds.reset()
        mds.fit_edges(batch.edge_index, trj_out[i].detach())
        trj.xyz = np.append(trj.xyz, [mds.pos.detach().cpu().numpy()], axis=0)
    trj.save(os.path.join('plots', f'{pdb_name}.pdb'))
    data_loader.train()


