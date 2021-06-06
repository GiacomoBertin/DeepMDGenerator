from typing import Optional, List, Union
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as func
from geomloss import SamplesLoss
from simtk.openmm import *
from torch import Tensor
from torch import nn
from torch.autograd import Variable
from torch.nn import BatchNorm1d, LayerNorm, InstanceNorm1d
from torch.nn import Parameter
from torch.nn import Sequential, Linear, Dropout
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import LayerNorm, BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import MessageNorm
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_scatter import scatter, scatter_softmax
from torch_sparse import SparseTensor

import utils.deep_utility as du
from utils.ModelAnalysis import ModelAnalysis
from utils.ProteinGraph import ProteinGraph
from utils.dataset import DeepMDDataset, DeepMDDataLoader

EPS = 1e-5
MAX_LOGSTD = 10
noise_dim = 6
latent_dim = 9


class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(torch.nn.ReLU())  # num_parameters=channels[i]
                m.append(Dropout(dropout))

        super(MLP, self).__init__(*m)


class PointConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, latent_dimension: int = 0,
                 aggr: str = 'softmax', t: float = 1.0, learn_t: bool = False,
                 p: float = 1.0, learn_p: bool = False, msg_norm: bool = False,
                 learn_msg_scale: bool = False, norm: str = 'batch', bias: bool = True,
                 num_layers: int = 4, eps: float = 1e-7, **kwargs):

        kwargs.setdefault('aggr', None)
        super(PointConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.eps = eps

        assert aggr in ['softmax', 'softmax_sg', 'power']

        channels = [2 * in_channels + latent_dimension]
        for i in range(num_layers - 1):
            channels.append(in_channels * 2)
        channels.append(out_channels)
        self.mlp_h = MLP(channels, norm=norm)
        self.mlp_p = MLP([3 * in_channels, 2 * in_channels, in_channels])
        # Linear(3 * in_channels, in_channels, bias=bias)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.initial_t = t
        self.initial_p = p

        if learn_t and aggr == 'softmax':
            self.t = Parameter(torch.Tensor([t]), requires_grad=True)
        else:
            self.t = t

        if learn_p:
            self.p = Parameter(torch.Tensor([p]), requires_grad=True)
        else:
            self.p = p

    def reset_parameters(self):
        reset(self.mlp_h)
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.msg_norm is not None:
            self.msg_norm.reset_parameters()
        if self.t and isinstance(self.t, Tensor):
            self.t.data.fill_(self.initial_t)
        if self.p and isinstance(self.p, Tensor):
            self.p.data.fill_(self.initial_p)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                latent_z: OptTensor = None, size: Size = None) -> Tuple[Tensor, Tensor]:
        """"""

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            if edge_attr is not None:
                assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()
            if edge_attr is not None:
                assert x[0].size(-1) == edge_attr.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        msg = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if self.msg_norm is not None:
            msg = self.msg_norm(x[0], msg)

        x_r = x[1]
        # if x_r is not None:
        #    out += x_r[:]
        if latent_z is not None:
            return self.mlp_h(torch.cat([x_r, msg, latent_z.unsqueeze(0).repeat(x_r.shape[0], 1)], dim=-1))
        else:
            return self.mlp_h(torch.cat([x_r, msg], dim=-1))

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_p(z)  # func.softplus(self.mlp_p(z))  # self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        if self.aggr == 'softmax':
            out = scatter_softmax(inputs * self.t, index, dim=self.node_dim)
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')

        elif self.aggr == 'softmax_sg':
            out = scatter_softmax(inputs * self.t, index,
                                  dim=self.node_dim).detach()
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')

        else:
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            return torch.pow(out, 1 / self.p)

    def __repr__(self):
        return '{}({}, {}, aggr={})'.format(self.__class__.__name__,
                                            self.in_channels,
                                            self.out_channels, self.aggr)


class GeneralConv(torch.nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 ckpt_grad=False,
                 latent_dimension: int = 0,
                 dropout=0.0,
                 norm=False):
        super(GeneralConv, self).__init__()
        self.act = nn.ReLU()
        self.edge_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * output_features + input_features, output_features * 2),
            nn.ReLU(),
            nn.Linear(output_features * 2, output_features),
        )
        self.conv_node = PointConv(in_channels=input_features, out_channels=output_features,
                                   latent_dimension=latent_dimension,
                                   learn_t=True)
        # self.rnn = GatedGraphConv(input_features, num_layers=3)
        self.ckpt_grad = ckpt_grad
        self.norm_in_h = BatchNorm(input_features)  # if norm else None
        self.norm_in_v = BatchNorm(input_features)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, latent_z: OptTensor = None):
        # Last 3 features: pos
        row, col = edge_index
        h = x
        v = edge_attr
        assert h.shape[1] == v.shape[1]
        if self.norm_in_h is not None:
            if self.ckpt_grad and h.requires_grad:
                h = checkpoint(self.norm_in_h, h)
            else:
                h = self.norm_in_h(h)

        h = self.act(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.norm_in_v is not None:
            if self.ckpt_grad and v.requires_grad:
                v = checkpoint(self.norm_in_v, v)
            else:
                v = self.norm_in_v(v)

        v = self.act(v)
        v = F.dropout(v, p=self.dropout, training=self.training)

        if self.ckpt_grad and h.requires_grad:
            # h = checkpoint(self.rnn, h, edge_index)
            h = checkpoint(self.conv_node, h, edge_index, v, latent_z)
            v = checkpoint(self.edge_mlp, torch.cat([h[row], h[col], v], 1))
        else:
            # h = self.rnn(h, edge_index)
            h = self.conv_node(h, edge_index, v, latent_z)
            v = self.edge_mlp(torch.cat([h[row], h[col], v], 1))

        return h, v


class PointBlock(nn.Module):
    def __init__(self,
                 input_features,
                 hiddens: List[int],
                 latent_dimension: int = 0,
                 ckpt_grad=False,
                 dropout=0.01):
        super(PointBlock, self).__init__()
        self.ckpt_grad = ckpt_grad
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = torch.nn.ModuleList()
        self.th = 2
        for i in range(len(hiddens)):
            from_feat = hiddens[i - 1] if i > 0 else input_features
            to_feat = hiddens[i]
            if i < self.th:
                self.layers.append(GeneralConv(from_feat, to_feat, latent_dimension=latent_dimension,
                                               ckpt_grad=True, dropout=dropout))
            else:
                self.layers.append(GeneralConv(from_feat, to_feat, ckpt_grad=True, dropout=dropout))

    def forward(self, x, edge_index, edge_attr, latent_z: OptTensor = None):
        h, v = x, edge_attr
        for i in range(0, len(self.layers)):
            if i < self.th:
                h, v = self.layers[i](h, edge_index, v, latent_z)
            else:
                h, v = self.layers[i](h, edge_index, v)
        return h  # self.edge_mlp(h)


class EncoderBlock(nn.Module):
    def __init__(self,
                 input_features,
                 hiddens: List[int],
                 ckpt_grad=False,
                 dropout=0.01):
        super(EncoderBlock, self).__init__()
        self.ckpt_grad = ckpt_grad
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = torch.nn.ModuleList()
        ratio = 0.7
        for i in range(len(hiddens)):
            from_feat = hiddens[i - 1] if i > 0 else input_features
            to_feat = hiddens[i]
            self.layers.append(GeneralConv(from_feat, to_feat, ckpt_grad=True, dropout=dropout))

    def forward(self, x, edge_index, edge_attr):
        h, out_edge_index, v = x, edge_index, edge_attr
        for i in range(0, len(self.layers)):
            h, v = self.layers[i](h, out_edge_index, v)
        h, _ = func.relu(h).max(dim=0)
        return torch.softmax(h + EPS, dim=0)


def edist(pos_ha, row, col):
    return torch.sqrt(torch.sum(torch.pow(pos_ha[row] - pos_ha[col], 2), dim=1))


class EdGE(nn.Module):
    def __init__(self,
                 input_features,
                 noise_dim,
                 latent_dim,
                 hiddens_en: List[int],
                 hiddens_ha: List[int],
                 ckpt_grad=True,
                 dropout=0.05):
        super(EdGE, self).__init__()
        self.ckpt_grad = ckpt_grad
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hiddens_en.append(latent_dim)
        self.encoder = EncoderBlock(input_features, hiddens_en, ckpt_grad=False)
        self.block_ha = PointBlock(input_features + 3,
                                   hiddens_ha, latent_dimension=(latent_dim + noise_dim),
                                   ckpt_grad=False)  # + latent_dim + 3, hiddens_ha)

    def forward(self, noise: Tensor, zero_pos: Tensor, data: ProteinGraph, recon=False) -> Tensor:
        # split in c_alpha, backbones, heavy atoms
        # HEAVY ATOMS
        row, col = data.edge_index
        # row, col = row.to(self.device), col.to(self.device)
        in_x, edge_index, in_edge_attr = data.x, data.edge_index, data.edge_attr
        # in_x, edge_index, in_edge_attr = in_x.to(self.device), edge_index.to(self.device), in_edge_attr.to(self.device)
        edges_length = torch.sqrt(torch.sum(torch.pow(zero_pos[row] - zero_pos[col], 2), dim=1))

        # The encoder has information about the starting configuration via edges_length
        if self.ckpt_grad:
            z = checkpoint(self.encoder, in_x, edge_index,
                           torch.cat((edges_length.unsqueeze(-1), in_edge_attr[:, 1:]), dim=-1))
        else:
            z = self.encoder(in_x, edge_index, torch.cat((edges_length.unsqueeze(-1), in_edge_attr[:, 1:]), dim=-1))
        # The next layers have only chemical information
        out_x = torch.cat((in_x,
                           torch.randn((in_x.shape[0], 3), device=self.device)
                           ), dim=1)
        ha_edge_attr = torch.cat((in_edge_attr[:, 1:],
                                  torch.randn((in_edge_attr.shape[0], 4), device=self.device)
                                  ), dim=1)

        if self.ckpt_grad:
            pos_ha = checkpoint(self.block_ha, out_x, edge_index, ha_edge_attr, torch.cat([z, noise], dim=0))
        else:
            pos_ha = self.block_ha(out_x, edge_index, ha_edge_attr, torch.cat([z, noise], dim=0))

        if recon:
            pos_ha = data.build_rings(pos_ha)

        del row, col, ha_edge_attr, out_x, z, edges_length, in_x, edge_index, in_edge_attr
        torch.cuda.empty_cache()
        return pos_ha

    def run(self, noises: Tensor, zero_pos: Tensor, data: ProteinGraph, save=False, recon=False):
        # split in c_alpha, backbones, heavy atoms
        # HEAVY ATOMS
        row, col = data.edge_index
        in_x, edge_index, in_edge_attr = data.x, data.edge_index, data.edge_attr
        edges_length = torch.sqrt(torch.sum(torch.pow(zero_pos[row] - zero_pos[col], 2), dim=1))

        # BACKBONE
        _, edge_bb_index, _, mask_bb_edge, mask_bb_nodes = du.get_subgraph(
            in_x, edge_index, in_edge_attr, 0, 1, renumber=True)

        trj_ha = []
        for i in range(noises.shape[0]):
            # The encoder has information about the starting configuration via edges_length
            if self.ckpt_grad:
                z = checkpoint(self.encoder, in_x, edge_index,
                               torch.cat((edges_length.unsqueeze(-1), in_edge_attr[:, 1:]), dim=-1))
            else:
                z = self.encoder(in_x, edge_index, torch.cat((edges_length.unsqueeze(-1), in_edge_attr[:, 1:]), dim=-1))
            # The next layers have only chemical information
            noise = noises[i]
            out_x = torch.cat((in_x,
                               torch.randn((in_x.shape[0], 3), device=self.device)
                               ), dim=1)
            ha_edge_attr = torch.cat((in_edge_attr[:, 1:],
                                      torch.randn((in_edge_attr.shape[0], 4), device=self.device)
                                      ), dim=1)
            # if self.ckpt_grad:
            #    pos_bb = checkpoint(self.block_bb, out_x[mask_bb_nodes], edge_bb_index, ha_edge_attr[mask_bb_edge],
            #                        torch.cat([z, noise], dim=0))
            # else:
            #    pos_bb = self.block_bb(out_x[mask_bb_nodes], edge_bb_index, ha_edge_attr[mask_bb_edge],
            #                           torch.cat([z, noise], dim=0))
            # out_x[mask_bb_nodes, -3:] = pos_bb

            if self.ckpt_grad:
                pos_ha = checkpoint(self.block_ha, out_x, edge_index, ha_edge_attr, torch.cat([z, noise], dim=0))
            else:
                pos_ha = self.block_ha(out_x, edge_index, ha_edge_attr, torch.cat([z, noise], dim=0))
            # pos_ha[mask_bb_nodes] = pos_bb
            if recon:
                pos_ha = data.build_rings(pos_ha)
            trj_ha.append(pos_ha)
            if self.ckpt_grad:
                edges_length = checkpoint(edist, pos_ha, row, col)
            else:
                edges_length = edist(pos_ha, row, col)

        trj_ha = torch.stack(trj_ha, dim=0)
        if save:
            trj_ha_recon = []
            for i in range(trj_ha.shape[0]):
                trj_ha_recon.append(data.build_rings(trj_ha[i]))
            trj_ha_recon = torch.stack(trj_ha_recon)
            data.save_dcd(trj_ha_recon.cpu().detach().numpy(), f'generated_pdb/{data.name}_gen_M.dcd')

        return trj_ha, mask_bb_nodes, mask_bb_edge


def compute_loss(pg: ProteinGraph, trj_hat, trj, bb_node_masks, epoch):
    mask_ha = pg.get_mask_tree_node()  # & torch.logical_not(pg.get_backbone_mask())
    loss_ha_b, loss_ha_a, loss_ha_d, = pg.compute_loss(trj_pos=trj_hat,
                                                       bond=True,
                                                       angle=True,
                                                       dihedral=True,
                                                       mask_nodes=mask_ha)

    loss_ha_ff = loss_ha_b
    loss_ha_ff += loss_ha_d if not torch.isnan(loss_ha_d) else 0.0
    loss_ha_ff += loss_ha_a if not torch.isnan(loss_ha_a) else 0.0
    ed = SamplesLoss("energy")

    def get_trj(t, t_hat, mask=None):
        mask_nb = pg.edge_attr[:, 1] == 0.

        if mask is not None:
            mask_edges = pg.get_mask_edges(mask) & mask_nb
        else:
            mask_edges = mask_nb

        t = torch.sqrt(torch.pow(t[:, pg.edge_index[0, mask_edges]] -
                                 t[:, pg.edge_index[1, mask_edges]], 2.0).sum(-1))
        t = du.reduce_edges_length(t.t()).t()
        t_hat = torch.sqrt(torch.pow(t_hat[:, pg.edge_index[0, mask_edges]] -
                                     t_hat[:, pg.edge_index[1, mask_edges]], 2.0).sum(-1))
        t_hat = du.reduce_edges_length(t_hat.t()).t()

        return ed(t, t_hat)

    def get_up(t, t_hat, mask_nodes=None):
        row, col = pg.edge_index

        d = torch.cdist(t, t)
        mask = torch.triu(torch.ones_like(d), diagonal=1) != 0
        mask_nb = torch.ones_like(mask, device=mask.device).to(torch.bool)
        mask_nb[:, row, col] = (pg.edge_attr[:, 1] == 0).repeat((mask.shape[0], 1))
        mask = mask_nb & mask

        d = torch.triu(d, diagonal=1)
        d = d[mask].view(d.shape[0], -1)

        d_hat = torch.cdist(t_hat, t_hat)
        mask = torch.triu(torch.ones_like(d_hat), diagonal=1) != 0
        mask_nb = torch.ones_like(mask, device=mask.device).to(torch.bool)
        mask_nb[:, row, col] = (pg.edge_attr[:, 1] == 0).repeat((mask.shape[0], 1))
        mask = mask_nb & mask

        d_hat = torch.triu(d_hat, diagonal=1)
        d_hat = d_hat[mask].view(d_hat.shape[0], -1)
        return ed(d, d_hat)

    def get_3n(t, t_hat, mask=None):
        t_a = ModelAnalysis.align_trj(t, torch.mean(t, dim=0), mask=mask)
        t_hat = ModelAnalysis.align_trj(t_hat, torch.mean(t, dim=0), mask=mask)
        return ed(t_a.view(t_a.shape[0], -1), t_hat.view(t_hat.shape[0], -1)), t_a, t_hat

    trj_ha_recon = []
    for i in range(trj_hat.shape[0]):
        trj_ha_recon.append(pg.build_rings(trj_hat[i]))
    trj_ha_recon = torch.stack(trj_ha_recon)
    # loss_bb_ed = get_trj(trj, trj_ha_recon, bb_node_masks)
    # loss_ha_ed = get_up(trj, trj_ha_recon)
    loss_ha_ed, t_a, t_hat = get_3n(trj, trj_ha_recon, bb_node_masks)
    # loss_bb_ed = ed(t_a[:, bb_node_masks].view(t_a.shape[0], -1), t_hat[:, bb_node_masks].view(t_hat.shape[0], -1))
    with open('train_loss.log', 'a') as file:
        file.write(f'{loss_ha_ed} {loss_ha_ff}\n')

    print(
        f'epoch: {epoch:>4d} '
        f' || loss_ha_ff {loss_ha_ff.item():>5.3f}'
        f' || loss_ha_ed {loss_ha_ed.item():>5.3f}'
        # f' || loss_bb_ed {loss_bb_ed.item():>5.3f}'
    )

    return loss_ha_ed / 0.02 + loss_ha_ff / 500.0


def analyze(model: EdGE, database: DeepMDDataset):
    man = ModelAnalysis(database)
    # man.model_run_pca_fin(model, dataset=database, noise_dim=noise_dim, n_runs=1000, pdb_name='pentapeptide',
    #                      out_prefix='out/penta')
    man.model_run_pca(model, dataset=database, noise_dim=noise_dim, n_runs=5000,
                      pdb_name='pentapeptide', out_prefix='out/penta', max_frame=database.max_frame)


def test(data_loader: DeepMDDataLoader, model: EdGE, noise_dim):
    losses = []
    data_loader.test()
    with torch.no_grad():
        for epoch, (pg, trj) in enumerate(data_loader):
            if epoch >= 20:
                break
            model.train()
            torch.cuda.empty_cache()

            noises = Variable(torch.randn((data_loader.test_set.trj_frames, noise_dim), device=data_loader.device))
            trj_ha, mask_bb_nodes, mask_bb_edge = \
                model.run(noises, trj[0], pg, save=(epoch >= 200) and (epoch % 50 == 0), recon=False)
            loss = compute_loss(pg, trj_ha, trj, mask_bb_nodes, epoch)
            losses.append(loss.item())
    data_loader.train()
    losses = np.array(losses)
    return losses.mean()


def train(dataset: DeepMDDataset, test_set: DeepMDDataset, model: EdGE, noise_dim):
    data_loader = DeepMDDataLoader(dataset=dataset, batch_size=1, test_set=test_set)
    model.to(data_loader.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    batch_size = data_loader.dataset.trj_frames
    batch_iter = 3
    best_loss = None
    best_train_loss = None
    file_ck = f'models/ckpt_model_{batch_size}_{data_loader.dataset.lag}_cing_M_3_CK.pt'
    file_train_ck = f'models/ckpt_model_{batch_size}_{data_loader.dataset.lag}_cing_M_BT_3_CK.pt'
    file = f'models/ckpt_model_{batch_size}_{data_loader.dataset.lag}_cing_M_3.pt'
    checkpoint_net = {'epoch': 0}
    if os.path.exists(file_ck):
        checkpoint_net = torch.load(file_ck)
        optimizer.load_state_dict(checkpoint_net['optimizer_state_dict'])
        model.load_state_dict(checkpoint_net['model_state_dict'])

    for epoch in range(checkpoint_net['epoch'], 100000000):
        model.train()
        data_loader.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        batch_trj = []
        batch_trj_hat = []
        mask_bb_nodes, mask_bb_edge, pg = None, None, None
        for b, (pg, trj) in enumerate(data_loader):
            if b >= batch_iter:
                break
            noises = Variable(torch.randn((batch_size, noise_dim), device=data_loader.device))
            trj_ha, mask_bb_nodes, mask_bb_edge = \
                model.run(noises, trj[0], pg, save=(epoch >= 50) and (epoch % 50 == 0), recon=False)
            batch_trj.append(trj)
            batch_trj_hat.append(trj_ha)

        loss = compute_loss(pg, torch.cat(batch_trj_hat, dim=0), torch.cat(batch_trj, dim=0), mask_bb_nodes,
                            epoch) / batch_iter

        if (best_train_loss is None) or (best_train_loss > loss):
            best_train_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, file_train_ck)

        if torch.isnan(loss):
            optimizer.zero_grad()
            # model.load_state_dict(torch.load('models/ckpt_model_501_1_ReLU_M_2.pt'))
            raise RuntimeError()

        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, file_ck)

        if epoch >= 3000 and epoch % 50 == 0:
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            md = ModelAnalysis()
            md.model_run_pca_fin(model, n_runs=10, noise_dim=noise_dim, dataset=data_loader.dataset,
                                 pdb_name='5awl', out_prefix='out/5awl')

        if epoch >= 200 and epoch % 50 == 0:
            test_loss = test(data_loader, model, noise_dim)
            print(f'{test_loss} {best_loss}')
            if (best_loss is None) or (best_loss > test_loss):
                best_loss = test_loss
                torch.save(model.state_dict(), file)


def run_main():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL"
    root_database = os.path.join(root, "train_cing")
    root_test = os.path.join(root, "train_cing")

    database = DeepMDDataset(root=root_database, lag=5, save_memory_mode=True, log_file='train.log')
    # pentapeptide has 5000 frames, each of 500 ns saved each 100 ps, MoDEL 10 ns saved each 1 ps
    # 100 * nanoseconds, 10 ps, 10'000
    # database.process(root_database)
    database.load_folder(root_database)
    database.get_trj = True
    database.max_frame = 1501
    database.trj_frames = int((database.max_frame - 1) / database.lag)

    test_set = DeepMDDataset(root=root_database, lag=database.lag, save_memory_mode=True, log_file='test.log')
    # pentapeptide has 5000 frames, each of 500 ns saved each 100 ps, MoDEL 10 ns saved each 1 ps
    # database.process(root_database)
    test_set.load_folder(root_database)
    test_set.get_trj = True
    test_set.trj_frames = 200

    model = EdGE(input_features=35, noise_dim=noise_dim, ckpt_grad=True, latent_dim=latent_dim,
                 hiddens_en=[60, 90, 60, 40, 15],
                 hiddens_ha=[60, 90, 120, 90, 60, 50, 30, 20, 3])
    train(database, test_set, model, noise_dim)


def run_test():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL"
    root_database = os.path.join(root, "train_cing")

    database = DeepMDDataset(root=root_database, lag=5, save_memory_mode=True, log_file='train.log')
    # pentapeptide has 5000 frames, each of 500 ns saved each 100 ps, MoDEL 10 ns saved each 1 ps
    # database.process(root_database)
    database.load_folder(root_database)
    database.get_trj = True
    database.max_frame = 1501
    database.trj_frames = int((database.max_frame - 1) / database.lag)

    model = EdGE(input_features=35, noise_dim=noise_dim, ckpt_grad=False, latent_dim=latent_dim,
                 hiddens_en=[60, 90, 60, 40, 15],
                 hiddens_ha=[60, 90, 120, 90, 60, 50, 30, 20, 3])
    batch_size = database.trj_frames
    file_ck = f'models/ckpt_model_{batch_size}_{database.lag}_cing_M_4_CK.pt'
    file_train_ck = f'models/ckpt_model_{batch_size}_{database.lag}_cing_M_BT_4_CK.pt'
    file = f'models/ckpt_model_{batch_size}_{database.lag}_cing_M_4.pt'

    checkpoint_net = torch.load(file_train_ck)
    model.load_state_dict(checkpoint_net['model_state_dict'])  # torch.load(file))#
    md = ModelAnalysis()
    model.to(model.device)
    md.model_run_pca_fin(model, n_runs=100, noise_dim=noise_dim, dataset=database,
                         pdb_name='5awl', out_prefix='model_analysis/5awl', lag=database.lag)


def run_energy_histo():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL"
    root_database = os.path.join(root, "train_cing")

    database = DeepMDDataset(root=root_database, lag=5, save_memory_mode=True, log_file='train.log')
    # pentapeptide has 5000 frames, each of 500 ns saved each 100 ps, MoDEL 10 ns saved each 1 ps
    # database.process(root_database)
    database.load_folder(root_database)
    database.get_trj = True
    database.max_frame = 1501
    database.trj_frames = int((database.max_frame - 1) / database.lag)
    pg, _ = next(database)
    md = ModelAnalysis()
    md.compute_energy_distribution(pg, database, ['all.dcd', 'deep.dcd'], 10000, labels=['Real', 'Generated'],
                                   lags=[10, 10], file='model_analysis/5awl')


# run_energy_histo()
# run_test()
# run_main()
# TODO: Perform a Montecarlo and compare net vs Montecarlo
