import os

import mdtraj as md
import numpy as np
import simtk.openmm as omm
import simtk.openmm.app as omm_app
import torch
from simtk.unit import *
from torch_geometric.data import Data
from torch_geometric.utils.num_nodes import maybe_num_nodes

from molecule import Molecule, Atom, amber_ff_atomstype, residues_encoded

RAD2DEG = 180 / np.pi

EPS = 1e-6
MAX_LOGSTD = 10

#EffeMuNu/GenGraphMSM

def find_rigid_alignment(A, B):
    """
    -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
    -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def get_TPR():
    positions = np.array(
        [[-2.48978604,  2.09017685, -0.00569446],
       [-3.60877222,  1.33617825, -0.36142383],
       [-3.47616246, -0.05181474, -0.430482  ],
       [-2.28903766, -0.65973779, -0.1591868 ],
       [-1.19017061,  0.09849565,  0.19136832],
       [-1.29453786,  1.4694058 ,  0.26680169],
       [-0.10392909, -0.72476508,  0.4209423 ],
       [-0.56328908, -2.00770991,  0.20275366],
       [-1.87021225, -1.92528359, -0.14042361],
       [ 1.2824363 , -0.39284247,  0.81360261],
       [ 2.0011519 ,  0.17223015, -0.57753083],
       [ 3.42083769,  0.44941524,  0.08571149],
       [ 4.33375142, -0.32654328,  0.1103089 ],
       [ 3.49197894,  1.74122374,  0.65814125],
       [ 2.35574101, -1.26842882, -1.07488868]]
    )
    cg_idx = 6
    cd_idx = [4, 7]
    ring_atoms = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    return positions, cg_idx, cd_idx, ring_atoms


def get_TYR():
    # import rdkit
    # from rdkit.Chem import AllChem
    # mol = rdkit.Chem.MolFromSmiles('c1cc(ccc1C[C@@H](C(=O)O)N)O')
    # AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    positions = [[-1.082, -0.730, -0.853],
                 [-2.352, -0.346, -1.215],
                 [-3.087, 0.468, -0.390],
                 [-2.480, 0.856, 0.784],
                 [-1.208, 0.473, 1.151],
                 [-0.470, -0.344, 0.321],
                 [-4.392, 0.891, -0.711]]
    dic = {'CG': [5], 'CD': [0, 4], 'CZ': [2], 'CE': [1, 3], 'OH': [6]}

    return dic, positions


def get_ij_pos(edge_idx, *args):
    row, col = edge_idx
    mask = (row == args[0]) & (col == args[1])
    return torch.masked_select(torch.arange(0, edge_idx.shape[1]), mask)[0]


def myacos(x):
    negate = (x < 0).to(torch.float)
    x = torch.abs(x)
    ret = -0.0187293
    ret = ret * x
    ret = ret + 0.0742610
    ret = ret * x
    ret = ret - 0.2121144
    ret = ret * x
    ret = ret + 1.5707288
    ret = ret * torch.sqrt(1.0 - x)
    ret = ret - 2 * negate * ret
    return negate * 3.14159265358979 + ret


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = vec1 / torch.norm(vec1), vec2 / torch.norm(vec2)
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = torch.norm(v)
    kmat = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = torch.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix


def compute_normal(pos):
    return (torch.linalg.svd((pos - pos.mean(dim=0, keepdim=True)).t())).U[:, -1]


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


def augment_edges_triangles_idx(tr_edges_idx: torch.Tensor):
    r = torch.zeros((2, tr_edges_idx.shape[1] * 2), dtype=torch.long, device=tr_edges_idx.device)
    a, b, c = tr_edges_idx.t()
    idx1 = torch.arange(0, tr_edges_idx.shape[1] * 2, 2, device=tr_edges_idx.device, dtype=torch.long)
    idx2 = idx1 + 1
    r[idx1, 0] = a
    r[idx1, 1] = b
    r[idx1, 2] = c
    r[idx2, 0] = a + 1
    r[idx2, 1] = b + 1
    r[idx2, 2] = c + 1
    return r


def reduce_edges(edges_index: torch.Tensor, tr_edges_idx, edges_length: torch.Tensor):
    idx1 = torch.arange(0, edges_index.shape[1], 2, device=edges_index.device, dtype=torch.long)
    idx2 = idx1 + torch.ones_like(idx1)
    le = (edges_length[idx1] + edges_length[idx2]) / 2.0
    e_idx = edges_index[:, idx1]
    n_idx = torch.zeros(edges_index.shape[1], dtype=torch.long, device=edges_index.device)
    n_idx[idx1] = torch.arange(0, e_idx.shape[1], device=edges_index.device, dtype=torch.long)
    n_idx[idx2] = torch.arange(0, e_idx.shape[1], device=edges_index.device, dtype=torch.long)
    tr = n_idx[tr_edges_idx]
    return e_idx, tr, le


def augment_edges_length(edges_length: torch.Tensor):
    if len(edges_length.shape) == 1:
        r = torch.zeros((edges_length.shape[0] * 2), dtype=edges_length.dtype, device=edges_length.device)
    else:
        r = torch.zeros((edges_length.shape[0] * 2, edges_length.shape[1]), dtype=edges_length.dtype,
                        device=edges_length.device)

    idx1 = torch.arange(0, edges_length.shape[0] * 2, 2, device=edges_length.device, dtype=torch.long)
    idx2 = idx1 + 1
    r[idx1] = edges_length
    r[idx2] = edges_length
    return r


class ProteinGraph(Data):
    def __init__(self, name, mol: Molecule = None, group: str = None):
        super(ProteinGraph, self).__init__()
        self.group = group
        if self.group is None:
            self.group = Atom.get_heavy_atoms_code()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_length = torch.tensor([])  # (n_edges)

        self.pos = None

        self.bonds_parameters = torch.tensor([])  # atoms bonded (n_bonds, 2)
        self.angles_parameters = torch.tensor([])  # atoms bonded (n_angles, 2)
        self.dihedrals_parameters = torch.tensor([])  # atoms bonded (n_dihedrals, 3)

        self.bonds_edges_indexes = torch.tensor([])  # atoms bonded (n_bonds) ij -> [ij]
        self.angles_edges_indexes = torch.tensor([])  # atoms with angles (n_angles, 3) ijk -> [jk] [ik] [ij]
        self.dihedrals_edges_indexes = torch.tensor([])  # atoms with dihedrals (n_dihedrals, 3, 3)

        self.bonds_nodes_indexes = torch.tensor([])  # atoms bonded (n_bonds, 2) ij
        self.angles_nodes_indexes = torch.tensor([])  # atoms with angles (n_angles, 3) ijk
        self.dihedrals_nodes_indexes = torch.tensor([])  # atoms with angles (n_dihedrals, 4) ijkl

        self.triangles_nodes_indexes = torch.tensor([])  # atoms in triangle (n_triangles, 3) ijk
        self.triangles_edges_indexes = torch.tensor([])  # ijk -> [jk] [ik] [ij]

        self.atoms_type = torch.tensor([])

        if mol is not None:
            self.name = mol.pdb_name
            self.folder = mol.folder
            self.initialize(mol)
            self.pdb_noh = mol.pdb_noh
        else:
            self.name = name
            self.pdb_noh = None

    def initialize(self, mol: Molecule, no_edge_indx=True):
        mol.pdb_noh, _ = mol.remove_from_pdb()
        mol.generate(group=self.group, generate_edges_bond=False)
        mol.complete_KNN_edges(k=10, group=self.group)
        mol.complete_edges(group=Atom.get_backbones_atoms_code())
        mol.atoms_slices(group=self.group)
        self.atoms_type = torch.tensor(mol.get_atoms_type_encoded(), device=self.device)
        data = mol.generate_graph(group=self.group)

        self.edge_index = data.edge_index.to(self.device)
        self.x = data.x.to(self.device)
        self.edge_attr = data.edge_attr.to(self.device)
        self.edge_length = self.edge_attr[:, 0] / 10.0  # A to nm
        self.edge_attr[:, 0] = 0.0

        self.bonds_parameters = torch.tensor(mol.bonds[:, -2:], device=self.device)
        self.bonds_nodes_indexes = torch.tensor(mol.bonds[:, :2], device=self.device)

        self.angles_parameters = torch.tensor(mol.angles[:, -2:], device=self.device)
        self.angles_nodes_indexes = torch.tensor(mol.angles[:, :3], device=self.device)

        self.dihedrals_parameters = torch.tensor(mol.dihedrals[:, -3:], device=self.device)
        self.dihedrals_nodes_indexes = torch.tensor(mol.dihedrals[:, :4], device=self.device)

        if no_edge_indx:
            return True
        self.node_to_edge_indx()
        return True

    def save(self, folder):
        root = os.path.join(folder, 'protein_graph')
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(self.x, os.path.join(root, 'x_{}.pt'.format(self.name)))
        torch.save(self.edge_index, os.path.join(root, 'edges_index_{}.pt'.format(self.name)))
        torch.save(self.edge_attr, os.path.join(root, 'edge_attr_{}.pt'.format(self.name)))

        torch.save(self.bonds_nodes_indexes, os.path.join(root, 'bonds_nodes_indexes_{}.pt'.format(self.name)))
        torch.save(self.bonds_edges_indexes, os.path.join(root, 'bonds_edges_indexes_{}.pt'.format(self.name)))
        torch.save(self.bonds_parameters, os.path.join(root, 'bonds_parameters_{}.pt'.format(self.name)))

        torch.save(self.angles_nodes_indexes, os.path.join(root, 'angles_nodes_indexes_{}.pt'.format(self.name)))
        torch.save(self.angles_edges_indexes, os.path.join(root, 'angles_edges_indexes_{}.pt'.format(self.name)))
        torch.save(self.angles_parameters, os.path.join(root, 'angles_parameters_{}.pt'.format(self.name)))

        torch.save(self.dihedrals_nodes_indexes, os.path.join(root, 'dihedrals_nodes_indexes_{}.pt'.format(self.name)))
        torch.save(self.dihedrals_edges_indexes, os.path.join(root, 'dihedrals_edges_indexes_{}.pt'.format(self.name)))
        torch.save(self.dihedrals_parameters, os.path.join(root, 'dihedrals_parameters_{}.pt'.format(self.name)))

        torch.save(self.triangles_nodes_indexes, os.path.join(root, 'triangles_nodes_indexes_{}.pt'.format(self.name)))
        torch.save(self.triangles_edges_indexes, os.path.join(root, 'triangles_edges_indexes_{}.pt'.format(self.name)))

        torch.save(self.atoms_type, os.path.join(root, 'atoms_type_{}.pt'.format(self.name)))

    def load(self, folder):
        root = os.path.join(folder, 'protein_graph')
        self.name = folder.split('/')[-1]
        if os.path.exists(os.path.join(folder, self.name + '_noh.pdb')):
            self.pdb_noh = os.path.join(folder, self.name + '_noh.pdb')

        self.x = torch.load(os.path.join(root, 'x_{}.pt'.format(self.name))).to(self.device)
        self.edge_index = torch.load(os.path.join(root, 'edges_index_{}.pt'.format(self.name))).to(self.device)
        self.edge_attr = torch.load(os.path.join(root, 'edge_attr_{}.pt'.format(self.name))).to(self.device)

        self.bonds_nodes_indexes = torch.load(os.path.join(root, 'bonds_nodes_indexes_{}.pt'.format(self.name))).to(
            self.device).to(torch.long)
        self.bonds_edges_indexes = torch.load(os.path.join(root, 'bonds_edges_indexes_{}.pt'.format(self.name))).to(
            self.device).to(torch.long)
        self.bonds_parameters = torch.load(os.path.join(root, 'bonds_parameters_{}.pt'.format(self.name))).to(
            self.device)

        self.angles_nodes_indexes = torch.load(os.path.join(root, 'angles_nodes_indexes_{}.pt'.format(self.name))).to(
            self.device).to(torch.long)
        self.angles_edges_indexes = torch.load(os.path.join(root, 'angles_edges_indexes_{}.pt'.format(self.name))).to(
            self.device).to(torch.long)
        self.angles_parameters = torch.load(os.path.join(root, 'angles_parameters_{}.pt'.format(self.name))).to(
            self.device)

        self.dihedrals_nodes_indexes = torch.load(
            os.path.join(root, 'dihedrals_nodes_indexes_{}.pt'.format(self.name))).to(self.device).to(torch.long)
        self.dihedrals_edges_indexes = torch.load(
            os.path.join(root, 'dihedrals_edges_indexes_{}.pt'.format(self.name))).to(self.device).to(torch.long)
        self.dihedrals_parameters = torch.load(os.path.join(root, 'dihedrals_parameters_{}.pt'.format(self.name))).to(
            self.device)

        self.triangles_nodes_indexes = torch.load(
            os.path.join(root, 'triangles_nodes_indexes_{}.pt'.format(self.name))).to(self.device).to(torch.long)
        self.triangles_edges_indexes = torch.load(
            os.path.join(root, 'triangles_edges_indexes_{}.pt'.format(self.name))).to(self.device).to(torch.long)

        self.atoms_type = torch.load(
            os.path.join(root, 'atoms_type_{}.pt'.format(self.name))).to(self.device).to(torch.long)

    def atoms_slices(self, column, value=1.0):
        self.x, self.edge_index, self.edge_attr, mask_edges, mask_nodes = self.get_subgraph(column, value, True)
        # renumber the edges
        n_idx = torch.zeros(mask_edges.size(0), dtype=torch.long, device=self.device)
        n_idx[mask_edges] = torch.arange(mask_edges.sum().item(), device=self.device)

        mask_bonds = self.get_mask_bonds(mask_nodes)
        mask_angles = self.get_mask_angles(mask_nodes)
        mask_dihedral = self.get_mask_dihedral(mask_nodes)

        self.bonds_edges_indexes = n_idx[self.bonds_edges_indexes[mask_bonds]]
        self.angles_edges_indexes = n_idx[self.angles_edges_indexes[mask_angles]]
        self.dihedrals_edges_indexes = n_idx[self.dihedrals_edges_indexes[mask_dihedral]]

        self.bonds_nodes_indexes = n_idx[self.bonds_nodes_indexes[mask_bonds]]
        self.angles_nodes_indexes = n_idx[self.angles_nodes_indexes[mask_angles]]
        self.dihedrals_nodes_indexes = n_idx[self.dihedrals_nodes_indexes[mask_dihedral]]

    def get_mask_edges(self, mask_nodes):
        return mask_nodes[self.edge_index[0]] & mask_nodes[self.edge_index[1]]

    def get_mask_triangles(self, mask_nodes):
        return mask_nodes[self.triangles_nodes_indexes[:, 0]] & \
               mask_nodes[self.triangles_nodes_indexes[:, 1]] & \
               mask_nodes[self.triangles_nodes_indexes[:, 2]]

    def get_mask_bonds(self, mask_nodes):
        mask_bonds = mask_nodes[self.bonds_nodes_indexes[:, 0]] & \
                     mask_nodes[self.bonds_nodes_indexes[:, 1]]
        return mask_bonds

    def get_mask_angles(self, mask_nodes):
        mask_angles = mask_nodes[self.angles_nodes_indexes[:, 0]] & \
                      mask_nodes[self.angles_nodes_indexes[:, 1]] & \
                      mask_nodes[self.angles_nodes_indexes[:, 2]]
        return mask_angles

    def get_mask_dihedral(self, mask_nodes):
        mask_dihedral = mask_nodes[self.dihedrals_nodes_indexes[:, 0]] & \
                        mask_nodes[self.dihedrals_nodes_indexes[:, 1]] & \
                        mask_nodes[self.dihedrals_nodes_indexes[:, 2]] & \
                        mask_nodes[self.dihedrals_nodes_indexes[:, 3]]
        return mask_dihedral

    def get_subgraph(self, column, value=1.0, relabel: bool = False, mask=None):
        mask_nodes = self.x[:, column] == value if mask is None else mask
        sub_edge_index, sub_edge_attr, mask_edges = subgraph(mask_nodes, self.edge_index, self.edge_attr,
                                                             relabel, self.x.shape[0])
        sub_x = self.x[torch.masked_select(torch.arange(self.x.shape[0]), mask_nodes)]
        return sub_x, sub_edge_index, sub_edge_attr, mask_edges, mask_nodes

    def save_pdb(self, pos, out_pdb=None):
        if out_pdb is None:
            trj: md.Trajectory = md.load_pdb(self.pdb_noh if not os.path.exists(self.pdb_noh[:-4] + '_pos.pdb') else
                                             self.pdb_noh[:-4] + '_pos.pdb')
        else:
            trj: md.Trajectory = md.load_pdb(self.pdb_noh if not os.path.exists(out_pdb) else out_pdb)

        trj.xyz = np.append(trj.xyz, [pos.detach().cpu().numpy()], axis=0)
        trj.save_pdb(out_pdb if out_pdb is not None else self.pdb_noh[:-4] + '_pos.pdb')

    def save_dcd(self, trj_np, out_dcd):
        trj: md.Trajectory = md.load_pdb(self.pdb_noh)

        trj.xyz = np.concatenate([trj.xyz, trj_np])
        trj.save_dcd(out_dcd)

    def regularize_triangles(self, edge_length, tr_edges_indexes=None):
        if tr_edges_indexes is None:
            tr_edges_indexes = self.triangles_edges_indexes
        _, triangles_edges_indexes, l = reduce_edges(self.edge_index, tr_edges_indexes, edge_length)
        EPS = 1.0e-2
        lr = 0.01
        cov = False
        for i in range(1000):
            tr_l, _ = torch.sort(l[triangles_edges_indexes], dim=1)
            u = torch.max(tr_l[:, 2] - (tr_l[:, 1] + tr_l[:, 0]))
            if u < 0.0:
                cov = True
                break
            eps = u / 3
            l_e = eps * torch.eye(l.shape[0], device=self.device) + l.repeat(l.shape[0], 1)
            tr_l, _ = torch.sort(l_e[:, triangles_edges_indexes], dim=2)
            u_e = torch.max(tr_l[:, :, 2] - (tr_l[:, :, 1] + tr_l[:, :, 0]), dim=1)[0]
            dude = (u_e - u) / eps
            l = l - lr * dude
        assert cov
        return augment_edges_length(l)

    def regularize_structure(self, edge_length=None):
        if self.pos is None:
            pos = torch.nn.Parameter(torch.randn((self.x.shape[0], 3), device=self.device), requires_grad=True)
        else:
            pos = torch.nn.Parameter(self.pos, requires_grad=True)
        optimizer = torch.optim.Adam([pos], lr=0.01)
        row, col = self.edge_index
        max_iter = 5000
        mse = torch.nn.MSELoss()
        epsilon = 1e-3
        for i in range(max_iter + 1):
            optimizer.zero_grad()
            d = torch.sqrt(torch.sum(torch.pow(pos[row] - pos[col], 2), dim=1))
            loss = mse(d, edge_length)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                # print(f' iter: {i:>5d} fit mse: {loss.item():>5.3f}')
                if loss.item() < epsilon:
                    print(f' iter: {i:>5d} fit mse: {loss.item():>5.3f}')
                    break

        self.pos = pos.data
        e = (edge_length - torch.sqrt(torch.sum(torch.pow(pos[row] - pos[col], 2), dim=1))).detach()
        return e, pos.data

    def get_ij_pos(self, *args):
        row, col = self.edge_index
        mask = torch.logical_and(row == args[0], col == args[1])
        return torch.masked_select(torch.arange(0, self.edge_index.shape[1]), mask)[0]

    def compute_triangles_edges_idx(self, triangles_nodes_idx, edge_idx):
        tr_edges = []
        for (a, b, c) in triangles_nodes_idx:
            tr_edges.append(
                [self.get_ij_pos(edge_idx, a, b), self.get_ij_pos(edge_idx, b, c), self.get_ij_pos(edge_idx, a, c)])
            tr_edges.append(
                [self.get_ij_pos(edge_idx, b, a), self.get_ij_pos(edge_idx, c, b), self.get_ij_pos(edge_idx, c, a)])
        tr_edges = torch.tensor(tr_edges)
        return tr_edges

    def compute_loss(self, trj_edges: torch.Tensor = None, trj_pos: torch.Tensor = None,
                     bond=True, angle=True, dihedral=True, mask_nodes=None):
        loss_bond = torch.tensor(0., device=self.device)
        loss_angles = torch.tensor(0., device=self.device)
        loss_dihedrals = torch.tensor(0., device=self.device)
        T = trj_edges.shape[0] if trj_pos is None else trj_pos.shape[0]
        N = self.edge_index.shape[1]
        for t in range(T):
            if trj_pos is None:
                edges_length = trj_edges[t]

                loss_bond += self.compute_bond_loss(mask_nodes=mask_nodes,
                                                    edge_length=edges_length) if bond else 0
                loss_angles += self.compute_angles_loss(mask_nodes=mask_nodes,
                                                        edge_length=edges_length) if angle else 0
                loss_dihedrals += self.compute_dihedral_loss(mask_nodes=mask_nodes,
                                                             edge_length=edges_length) if dihedral else 0
            else:
                pos = trj_pos[t]

                loss_bond += self.compute_bond_loss(mask_nodes=mask_nodes, pos=pos) if bond else 0
                loss_angles += self.compute_angles_loss(mask_nodes=mask_nodes, pos=pos) if angle else 0
                loss_dihedrals += self.compute_dihedral_loss(mask_nodes=mask_nodes, pos=pos) if dihedral else 0

        return loss_bond / T, loss_angles / T, loss_dihedrals / T

    def compute_bond_loss(self, edge_length=None, pos=None, mask_nodes=None):
        # i---j
        if pos is None:
            if edge_length is None:
                edge_length = self.edge_length

            if mask_nodes is not None:
                mask = self.get_mask_bonds(mask_nodes)
                loss = self.bonds_parameters[mask, 1] * torch.pow(
                    edge_length[self.bonds_edges_indexes[mask]] - self.bonds_parameters[mask, 0], 2.0)
            else:
                loss = self.bonds_parameters[:, 1] * torch.pow(
                    edge_length[self.bonds_edges_indexes] - self.bonds_parameters[:, 0], 2.0)
        else:
            if mask_nodes is not None:
                mask = self.get_mask_bonds(mask_nodes)
                length = torch.sqrt(torch.pow(
                    pos[self.bonds_nodes_indexes[mask, 0]] - pos[self.bonds_nodes_indexes[mask, 1]],
                    2.0).sum(-1))
                loss = self.bonds_parameters[mask, 1] * torch.pow(length - self.bonds_parameters[mask, 0], 2.0)
            else:
                length = torch.sqrt(torch.pow(
                    pos[self.bonds_nodes_indexes[:, 0]] - pos[self.bonds_nodes_indexes[:, 1]],
                    2.0).sum(-1))
                loss = self.bonds_parameters[:, 1] * torch.pow(length - self.bonds_parameters[:, 0], 2.0)
        return loss.mean()

    @staticmethod
    def are_triangles(edge_length, triangles_edges_indexes):
        tr_l, _ = torch.sort(edge_length[triangles_edges_indexes], dim=1)
        u = torch.max(tr_l[:, 2] - (tr_l[:, 1] + tr_l[:, 0]))
        if u > 0:
            return False
        else:
            return True

    def compute_angles_loss(self, edge_length=None, pos=None, mask_nodes=None):
        #    i
        #    |
        #    J
        #     \
        #      K
        if pos is None:
            if mask_nodes is not None:
                mask = self.get_mask_bonds(mask_nodes)
                if edge_length is None:
                    edge_length = self.edge_length
                cos_angles_ijk = (torch.pow(edge_length[self.angles_edges_indexes[mask, 0]], 2) +
                                  torch.pow(edge_length[self.angles_edges_indexes[mask, 2]], 2) -
                                  torch.pow(edge_length[self.angles_edges_indexes[mask, 1]], 2)) / \
                                 (2 * edge_length[self.angles_edges_indexes[mask, 0]] * edge_length[
                                     self.angles_edges_indexes[mask, 2]])
                corr = (torch.logical_or(cos_angles_ijk < -1.0, cos_angles_ijk > 1.0)).to(torch.long).sum()
                if corr > 0:
                    print(cos_angles_ijk)
                angles_ijk = torch.arccos(cos_angles_ijk)
                loss = self.angles_parameters[mask, 1] * torch.pow(angles_ijk -
                                                                   self.angles_parameters[mask, 0] * np.pi / 180.0, 2.0)
            else:
                if edge_length is None:
                    edge_length = self.edge_length
                cos_angles_ijk = (torch.pow(edge_length[self.angles_edges_indexes[:, 0]], 2) +
                                  torch.pow(edge_length[self.angles_edges_indexes[:, 2]], 2) -
                                  torch.pow(edge_length[self.angles_edges_indexes[:, 1]], 2)) / \
                                 (2 * edge_length[self.angles_edges_indexes[:, 0]] * edge_length[
                                     self.angles_edges_indexes[:, 2]])
                corr = (torch.logical_or(cos_angles_ijk < -1.0, cos_angles_ijk > 1.0)).to(torch.long).sum()
                if corr > 0:
                    print(cos_angles_ijk)
                angles_ijk = torch.arccos(cos_angles_ijk)
                loss = self.angles_parameters[:, 1] * torch.pow(angles_ijk -
                                                                self.angles_parameters[:, 0] * np.pi / 180.0, 2.0)
        else:
            if mask_nodes is not None:
                mask = self.get_mask_angles(mask_nodes)
                ij = pos[self.angles_nodes_indexes[mask, 1]] - pos[self.angles_nodes_indexes[mask, 0]]
                jk = pos[self.angles_nodes_indexes[mask, 2]] - pos[self.angles_nodes_indexes[mask, 1]]
                cos_angles_ijk = torch.sum(ij * jk, dim=1) / (torch.norm(ij, dim=1) * torch.norm(jk, dim=1) + EPS)
                corr = (torch.logical_or(cos_angles_ijk < -1.0, cos_angles_ijk > 1.0)).to(torch.long).sum()
                if corr > 0:
                    print(cos_angles_ijk)
                angles_ijk = torch.arccos(cos_angles_ijk)

                loss = self.angles_parameters[mask, 1] * torch.pow(angles_ijk -
                                                                   self.angles_parameters[mask, 0] / RAD2DEG, 2.0)
            else:
                ij = pos[self.angles_nodes_indexes[:, 1]] - pos[self.angles_nodes_indexes[:, 0]]
                jk = pos[self.angles_nodes_indexes[:, 2]] - pos[self.angles_nodes_indexes[:, 1]]
                cos_angles_ijk = torch.sum(ij * jk, dim=1) / (torch.norm(ij, dim=1) * torch.norm(jk, dim=1) + EPS)
                corr = (torch.logical_or(cos_angles_ijk < -1.0, cos_angles_ijk > 1.0)).to(torch.long).sum()
                if corr > 0:
                    print(cos_angles_ijk)
                cos_angles_ijk = torch.clamp(cos_angles_ijk, -1.0 + EPS, 1.0 - EPS)
                angles_ijk = torch.arccos(cos_angles_ijk)

                loss = self.angles_parameters[:, 1] * torch.pow(angles_ijk -
                                                                self.angles_parameters[:, 0] / RAD2DEG, 2.0)
        return loss.mean()

    def compute_dihedral_loss(self, edge_length=None, pos=None, mask_nodes=None):
        #   planes:          i         l
        #   ijk, jkl        /          |
        #                  j           i ___ k
        #                  |    or    /
        #                  k         j
        #                   \
        #                     l
        # cos(<ijkl) = (cos(<ijl) - cos(<ijk) cos(<ljk)) / (sin(<ijk) sin(<ljk))
        # [[ij] [jl] [il]] (ijl)
        # [[ij] [jk] [ik]] (ijk)
        # [[lj] [jk] [kl]] (ljk)
        if pos is None:
            if edge_length is None:
                edge_length = self.edge_length
            ijl = edge_length[self.dihedrals_edges_indexes[:, 0]]

            cos_angles_ijl = (torch.pow(ijl[:, 0], 2) +
                              torch.pow(ijl[:, 1], 2) -
                              torch.pow(ijl[:, 2], 2)) / (2 * ijl[:, 0] * ijl[:, 1])  # il

            ijk = edge_length[self.dihedrals_edges_indexes[:, 1]]

            cos_angles_ijk = (torch.pow(ijk[:, 0], 2) +  # ik
                              torch.pow(ijk[:, 1], 2) -
                              torch.pow(ijk[:, 2], 2)) / (2 * ijk[:, 0] * ijk[:, 1])

            ljk = edge_length[self.dihedrals_edges_indexes[:, 2]]

            cos_angles_ljk = (torch.pow(ljk[:, 0], 2) +  # ik
                              torch.pow(ljk[:, 1], 2) -
                              torch.pow(ljk[:, 2], 2)) / (2 * ljk[:, 0] * ljk[:, 1])

            dihedral = torch.arccos((cos_angles_ijl - cos_angles_ijk * cos_angles_ljk) /
                                    (torch.sqrt(1.0 - torch.pow(cos_angles_ijk, 2)) *
                                     torch.sqrt(1.0 - torch.pow(cos_angles_ljk, 2))))

            loss = self.dihedrals_parameters[:, 1] * (
                    1.0 + torch.cos(self.dihedrals_parameters[:, 2] * dihedral * 180 / np.pi -
                                    self.dihedrals_parameters[:, 0]))
        else:
            if mask_nodes is not None:
                mask = self.get_mask_dihedral(mask_nodes)

                a1 = pos[self.dihedrals_nodes_indexes[mask, 1]] - pos[self.dihedrals_nodes_indexes[mask, 0]]
                # coords2 - coords1
                a2 = pos[self.dihedrals_nodes_indexes[mask, 2]] - pos[self.dihedrals_nodes_indexes[mask, 1]]
                # coords3 - coords2
                a3 = pos[self.dihedrals_nodes_indexes[mask, 3]] - pos[self.dihedrals_nodes_indexes[mask, 2]]
                # coords4 - coords3

                v1 = torch.cross(a1, a2)
                v1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
                v2 = torch.cross(a2, a3)
                v2 = v2 / torch.linalg.norm(v1, dim=-1, keepdim=True)
                porm = torch.sign((v1 * a3).sum(-1))
                rad = torch.arccos((v1 * v2).sum(-1))
                rad[porm != 0.0] = rad[porm != 0.0] * porm[porm != 0.0]
                loss = self.dihedrals_parameters[mask, 1] * (
                        1.0 + torch.cos(self.dihedrals_parameters[mask, 2] * rad * RAD2DEG -
                                        self.dihedrals_parameters[mask, 0]))
            else:
                a1 = pos[self.dihedrals_nodes_indexes[:, 1]] - pos[
                    self.dihedrals_nodes_indexes[:, 0]]  # coords2 - coords1
                a2 = pos[self.dihedrals_nodes_indexes[:, 2]] - pos[
                    self.dihedrals_nodes_indexes[:, 1]]  # coords3 - coords2
                a3 = pos[self.dihedrals_nodes_indexes[:, 3]] - pos[
                    self.dihedrals_nodes_indexes[:, 2]]  # coords4 - coords3

                v1 = torch.cross(a1, a2)
                v1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)
                v2 = torch.cross(a2, a3)
                v2 = v2 / torch.linalg.norm(v1, dim=-1, keepdim=True)
                porm = torch.sign((v1 * a3).sum(-1))
                rad = torch.arccos((v1 * v2).sum(-1))
                rad[porm != 0.0] = rad[porm != 0.0] * porm[porm != 0.0]

                loss = self.dihedrals_parameters[:, 1] * (
                        1.0 + torch.cos(self.dihedrals_parameters[:, 2] * rad * RAD2DEG -
                                        self.dihedrals_parameters[:, 0]))

        return loss.mean()

    def node_to_edge_indx(self):

        self.bonds_edges_indexes = []
        for (a, b) in self.bonds_nodes_indexes:
            ab = get_ij_pos(self.edge_index, a, b)
            # ba = ab + 1
            self.bonds_edges_indexes.append(ab)
            # self.bonds_edges_indexes.append(ba)
        self.bonds_edges_indexes = torch.tensor(self.bonds_edges_indexes)
        self.triangles_nodes_indexes = []
        self.triangles_edges_indexes = []

        self.angles_edges_indexes = []
        for (a, b, c) in self.angles_nodes_indexes:
            ab = get_ij_pos(self.edge_index, a, b)
            bc = get_ij_pos(self.edge_index, b, c)
            ca = get_ij_pos(self.edge_index, c, a)

            self.angles_edges_indexes.append([bc, ca, ab])
            self.triangles_nodes_indexes.append([a, b, c])
            self.triangles_edges_indexes.append([bc, ca, ab])
            # self.angles_edges_indexes.append([cb, ac, ba])
        self.angles_edges_indexes = torch.tensor(self.angles_edges_indexes)

        # ijkl ->  cos d = (cos(ijl) - cos(ijk) cos(ljk)) / (sin(ijk) sin(ljk))
        # [[ij] [jl] [il]] (ijl)
        # [[ij] [jk] [ik]] (ijk)
        # [[lj] [jk] [kl]] (ljk)

        self.dihedrals_edges_indexes = []
        for (i, j, k, l) in self.dihedrals_nodes_indexes:
            ij = get_ij_pos(self.edge_index, i, j)
            jl = get_ij_pos(self.edge_index, j, l)
            il = get_ij_pos(self.edge_index, i, l)

            jk = get_ij_pos(self.edge_index, j, k)
            # ij
            ik = get_ij_pos(self.edge_index, i, k)

            lj = get_ij_pos(self.edge_index, l, j)
            # jk = ki
            kl = get_ij_pos(self.edge_index, k, l)

            self.dihedrals_edges_indexes.append([[ij, jl, il], [ij, jk, ik], [lj, jk, kl]])

            self.triangles_nodes_indexes.append([i, j, l])
            self.triangles_nodes_indexes.append([i, j, k])
            self.triangles_nodes_indexes.append([l, j, k])

            self.triangles_edges_indexes.append([ij, jl, il])
            self.triangles_edges_indexes.append([ij, jk, ik])
            self.triangles_edges_indexes.append([lj, jk, kl])

        self.dihedrals_edges_indexes = torch.tensor(self.dihedrals_edges_indexes)
        self.triangles_edges_indexes = torch.tensor(self.triangles_edges_indexes)
        self.triangles_nodes_indexes = torch.tensor(self.triangles_nodes_indexes)

    def minimize(self, pos_numpy):
        solute_dielectric = 6.0
        solvent_dielectric = 78.5
        prmtop_file = os.path.join(self.folder, self.name, f'{self.name}.prmtop')
        prmcrd_file = os.path.join(self.folder, self.name, f'{self.name}.prmcrd')
        picosecond = pico * second
        nanometer = nano * meter

        # ONLY PROTEIN
        prmtop = omm_app.AmberPrmtopFile(prmtop_file)
        inpcrd = omm_app.AmberInpcrdFile(prmcrd_file, loadBoxVectors=True)
        system = prmtop.createSystem(nonbondedMethod=omm_app.CutoffNonPeriodic, nonbondedCutoff=1 * nanometer,
                                     constraints=omm_app.HBonds,
                                     implicitSolvent=omm_app.OBC2, soluteDielectric=solute_dielectric,
                                     solventDielectric=solvent_dielectric,
                                     implicitSolventSaltConc=0.15 * molar)

        system.addForce(omm.AndersenThermostat(298.15, 1.0))
        integrator = omm.VerletIntegrator(0.002 * picosecond)
        my_simulation = omm_app.Simulation(prmtop.topology, system, integrator)
        if inpcrd.boxVectors is not None:
            my_simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
        my_simulation.context.setPositions(pos_numpy)
        my_simulation.minimizeEnergy(maxIterations=10000, tolerance=0.5 * kilocalorie_per_mole)
        state: omm.State = my_simulation.context.getState(getPositions=True)

        return state.getPositions(asNumpy=True)

    def build_rings(self, pos):
        assert pos.shape[-2] == self.x.shape[0]
        residues = self.x[:, 4].to(torch.long)
        atoms_type = self.atoms_type
        for i in range(1, residues.max().item() + 1):

            # Compute the average point and normal plane
            mask = (self.x[:, 4] == i) & (self.x[:, 3] == 1.0)

            if mask.to(torch.long).sum() == 0:
                continue
            mask_CG_CD = mask & ((self.atoms_type == amber_ff_atomstype.index('CG')) |
                                 (self.atoms_type == amber_ff_atomstype.index('CD')))

            if mask.to(torch.long).sum() == 5:
                continue
            elif mask.to(torch.long).sum() == 6:
                mask_CG = mask & (atoms_type == amber_ff_atomstype.index('CG'))
                dic, pos_tyr = get_TYR()
                pos_tyr = torch.tensor(pos_tyr, device=pos.device).float() / 10.0
                cg_idx = torch.tensor(dic['CG'], device=pos.device).to(torch.long)
                cd_idx = torch.tensor(dic['CD'], device=pos.device).to(torch.long)
                ce_idx = torch.tensor(dic['CE'], device=pos.device).to(torch.long)
                cz_idx = torch.tensor(dic['CZ'], device=pos.device).to(torch.long)
                oh_idx = torch.tensor(dic['OH'], device=pos.device).to(torch.long)
                rot, t = find_rigid_alignment(pos_tyr[torch.cat([cg_idx, cd_idx], dim=0)], pos[mask_CG_CD])
                pos_tyr = (rot.mm(pos_tyr.T)).T + t
                shift = pos[mask_CG] - pos_tyr[cg_idx]
                pos[mask & (atoms_type == amber_ff_atomstype.index('CG'))] = (pos_tyr[cg_idx] + shift)
                pos[mask & (atoms_type == amber_ff_atomstype.index('CD'))] = (pos_tyr[cd_idx] + shift)
                pos[mask & (atoms_type == amber_ff_atomstype.index('CE'))] = (pos_tyr[ce_idx] + shift)
                pos[mask & (atoms_type == amber_ff_atomstype.index('CZ'))] = (pos_tyr[cz_idx] + shift)
                pos[(self.x[:, 4] == i) & (atoms_type == amber_ff_atomstype.index('OH'))] = (pos_tyr[oh_idx] + shift)
            elif mask.to(torch.long).sum() > 6:
                mask_CG = mask & (atoms_type == amber_ff_atomstype.index('CG'))

                tpr, cg_idx, cd_idx, ring_idx = get_TPR()
                tpr, cg_idx, cd_idx, ring_idx = torch.tensor(tpr, device=self.device).float() / 10.0, \
                                                torch.tensor(cg_idx, device=self.device).to(torch.long), \
                                                torch.tensor(cd_idx, device=self.device).to(torch.long), \
                                                torch.tensor(ring_idx, device=self.device).to(torch.long)

                rot, t = find_rigid_alignment(tpr[torch.cat((cg_idx.view(1), cd_idx))], pos[mask_CG_CD])
                #rot, t = find_rigid_alignment(tpr[ring_idx], pos[mask])
                tpr = (rot.mm(tpr[ring_idx].T)).T + t
                # Somehow works... but sucks

                shift = pos[mask_CG] - tpr[cg_idx]
                pos[mask & (atoms_type == amber_ff_atomstype.index('CZ'))] = torch.stack(
                    ((tpr[ring_idx[2]] + shift).squeeze(0),
                     (tpr[ring_idx[0]] + shift).squeeze(0)), dim=0)
                pos[mask & (atoms_type == amber_ff_atomstype.index('CH'))] = (tpr[ring_idx[1]] + shift)
                pos[mask & (atoms_type == amber_ff_atomstype.index('CE'))] = torch.stack(
                    ((tpr[ring_idx[3]] + shift).squeeze(0),
                     (tpr[ring_idx[5]] + shift).squeeze(0)), dim=0)
                pos[mask & (atoms_type == amber_ff_atomstype.index('CD'))] = torch.stack(
                    ((tpr[ring_idx[7]] + shift).squeeze(0),
                     (tpr[ring_idx[4]] + shift).squeeze(0)), dim=0)
                pos[mask & (atoms_type == amber_ff_atomstype.index('CG'))] = (tpr[ring_idx[6]] + shift)
                pos[mask & (atoms_type == amber_ff_atomstype.index('NE'))] = (tpr[ring_idx[8]] + shift)
            else:
                continue

        return pos

    def flat_rings(self, pos):
        # Identify rings atoms
        assert pos.shape[-2] == self.x.shape[0]
        residues = self.x[:, 4].to(torch.long)

        for i in range(residues.max().item()):

            # Compute the average point and normal plane
            mask = (self.x[:, 4] == residues[i]) & (self.x[:, 3] == 1.0)
            if mask.to(torch.long).sum() == 0:
                continue
            mean = pos[mask].mean(dim=0, keepdim=True)
            points = pos[mask] - mean
            norm = (torch.linalg.svd(points.t())).U[:, -1]

            # Project
            pos[mask] = pos[mask] - ((pos[mask] - mean) * norm).sum(dim=-1, keepdim=True) * norm
        return pos

    def get_backbone_mask(self):
        return self.x[:, 0] == 1.0

    def get_ca_mask(self):
        return self.x[:, 1] == 1.0

    def get_atoms_mask(self, name):
        return self.atoms_type == amber_ff_atomstype.index(name)

    def get_mask_tree_node(self):
        # Keep only CG, CD1, CD2 for each ring
        mask = (self.x[:, 3] != 1.0) |\
               (self.atoms_type == amber_ff_atomstype.index('CG')) | \
               (self.atoms_type == amber_ff_atomstype.index('CD')) | \
               (self.x[:, 11 + residues_encoded.index('PRO')] == 1.0) | \
                self.get_backbone_mask()
        return mask
