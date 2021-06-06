import numpy as np
import torch
from torch_scatter import scatter_mean
from simtk.openmm import *
from simtk.openmm.app import *


def get_ij_pos(edge_idx, *args):
    row, col = edge_idx
    mask = torch.logical_and(row == args[0], col == args[1])
    return torch.masked_select(torch.arange(0, edge_idx.shape[1]), mask)[0]


def triangles_idx(tr_idx, edge_idx):
    tr_edges = []
    for (a, b, c) in tr_idx:
        tr_edges.append([get_ij_pos(edge_idx, a, b), get_ij_pos(edge_idx, b, c), get_ij_pos(edge_idx, a, c)])
    tr_edges = torch.tensor(tr_edges)
    return tr_edges


def triangles_loss(tr_idx, edges_l):
    edges, _ = torch.sort(edges_l[tr_idx.view(-1)].view(-1, tr_idx.shape[1]), dim=1)
    return torch.max((edges[:, 2] - (edges[:, 1] + edges[:, 0])))


def regularize_edges(tr_idx, edges_l):
    # par = torch.nn.Parameter(edges_l, requires_grad=True)
    optimizer = torch.optim.Adam([edges_l], lr=0.005)
    for i in range(10000):
        optimizer.zero_grad()
        loss = triangles_loss(tr_idx, edges_l)
        if loss < 0:
            break
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 100 == 0:
            print(i, loss.item())


def regularize_edges1(tr_idx, edges_l):
    l = edges_l
    EPS = 1.0 + 1.0e-3
    for i in range(1000):
        edges, idx = torch.sort(l[tr_idx], dim=1)
        e = edges[:, 2] - (edges[:, 1] + edges[:, 0])
        mask = e > 0
        ee = torch.zeros_like(edges)
        ee[mask, 0] = +e[mask] * edges[mask, 0] / torch.sum(edges, dim=1)
        ee[mask, 1] = +e[mask] * edges[mask, 1] / torch.sum(edges, dim=1) * EPS
        ee[mask, 2] = -e[mask] * edges[mask, 2] / torch.sum(edges, dim=1) * EPS
        l = l + scatter_mean(ee.view(-1), tr_idx.view(-1))


"""indexes = []
l = []
for i in range(0, 4):
    for j in range(i + 1, 4):
        indexes.append([i, j])
        indexes.append([j, i])
        t = np.random.randn()
        l.append(abs(t))
        l.append(abs(t))

triangles = [[0, 1, 3], [0, 1, 2], [1, 2, 3]]
tr_idx = torch.tensor(triangles)
edge_idx = torch.tensor(indexes).t()
edge_len = torch.tensor(l, requires_grad=True)
tr_edg_idx = triangles_idx(tr_idx, edge_idx)

l = edge_len
EPS = 1.0e-2
lr = 0.01

for i in range(1000):
    edges, _ = torch.sort(l[tr_edg_idx], dim=1)
    u = torch.max(edges[:, 2] - (edges[:, 1] + edges[:, 0]))
    print(u)
    if u.item() < 0:
        print(edge_len, l)
        break
    l_e = EPS * torch.eye(l.shape[0]) + l.repeat(l.shape[0], 1)
    tr_l, _ = torch.sort(l_e[:, tr_edg_idx], dim=2)
    u_e = torch.max(tr_l[:, :, 2] - (tr_l[:, :, 1] + tr_l[:, :, 0]), dim=1)[0]
    dude = (u_e - u) / EPS
    l = l - lr * dude"""


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
    return rotation_matrix.float()


def compute_normal(pos):
    return (torch.linalg.svd((pos - pos.mean(dim=0, keepdim=True)).t())).U[:, -1]


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


def get_TPR():
    import rdkit
    from rdkit.Chem import AllChem
    mol = rdkit.Chem.MolFromSmiles('c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N')
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    ring_atoms = []
    cg_idx = None
    cd_idx = []
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            ring_atoms.append(atom.GetIdx())
            is_not_cg = True
            for bond in atom.GetBonds():
                is_not_cg = is_not_cg and bond.GetEndAtom().IsInRing()
            if not is_not_cg:
                cg_idx = atom.GetIdx()
                for bond in atom.GetBonds():
                    if bond.GetEndAtom().IsInRing() and bond.GetEndAtomIdx() != cg_idx:
                        cd_idx.append(bond.GetEndAtomIdx())
                    elif bond.GetBeginAtom().IsInRing() and bond.GetBeginAtomIdx() != cg_idx:
                        cd_idx.append(bond.GetBeginAtomIdx())

    return mol, mol.GetConformer().GetPositions(), cg_idx, cd_idx, ring_atoms


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


def build_rings():
    from molecule import amber_ff_atomstype
    import mdtraj as md
    atoms_type = []  # TODO
    pdb = 'generated_pdb/test_TPR.pdb'
    x = torch.load('./generated_pdb/x_pentapeptide.pt').cpu()
    residues = x[:, 4]
    residues = residues.to(torch.long)

    with open(pdb) as file:
        for line in file:
            if line.startswith('ATOM'):
                t = line.split()[2][:2]
                if ('C' in t) and (t not in amber_ff_atomstype):
                    t = 'C*'
                atoms_type.append(amber_ff_atomstype.index(t))
    atoms_type = torch.tensor(atoms_type)
    pos = torch.tensor(md.load_pdb(pdb).xyz[0])
    for i in range(int(residues.max().item())):

        # Compute the average point and normal plane
        mask = (x[:, 4] == residues[i]) & (x[:, 3] == 1.0)

        if mask.to(torch.long).sum() == 0:
            continue

        mean = pos[mask].mean(dim=0, keepdim=True)
        points = pos[(atoms_type == amber_ff_atomstype.index('CG')) |
                     (atoms_type == amber_ff_atomstype.index('CB')) |
                     (atoms_type == amber_ff_atomstype.index('CD'))] - mean
        norm = (torch.linalg.svd(points.t())).U[:, -1].float()

        if mask.to(torch.long).sum() == 5:
            mask_CG = mask & (atoms_type == amber_ff_atomstype.index('CG'))
            dic, pos_tyr = get_TYR()
            pos_tyr = torch.tensor(pos_tyr).float() / 10.0
            cg_idx = torch.tensor(dic['CG']).to(torch.long)
            cd_idx = torch.tensor(dic['CD']).to(torch.long)
            ce_idx = torch.tensor(dic['CE']).to(torch.long)
            cz_idx = torch.tensor(dic['CZ']).to(torch.long)
            oh_idx = torch.tensor(dic['OH']).to(torch.long)
            rot, t = find_rigid_alignment(pos_tyr[:-1], pos[mask])
            pos_tyr = (rot.mm(pos_tyr.T)).T + t
            shift = pos[mask_CG] - pos_tyr[cg_idx]
            pos[mask & (atoms_type == amber_ff_atomstype.index('CG'))] = (pos_tyr[cg_idx] + shift)
            pos[mask & (atoms_type == amber_ff_atomstype.index('CD'))] = (pos_tyr[cd_idx] + shift)
            pos[mask & (atoms_type == amber_ff_atomstype.index('CE'))] = (pos_tyr[ce_idx] + shift)
            pos[mask & (atoms_type == amber_ff_atomstype.index('CZ'))] = (pos_tyr[cz_idx] + shift)
            pos[mask & (atoms_type == amber_ff_atomstype.index('OH'))] = (pos_tyr[oh_idx] + shift)

        elif mask.to(torch.long).sum() == 6:
            pass
        elif mask.to(torch.long).sum() > 6:
            renumber = torch.tensor([6, 5, 4, 3, 8, 7, 0, 1, 2])
            mask_CG = mask & (atoms_type == amber_ff_atomstype.index('CG'))

            _, tpr, cg_idx, ring_idx = get_TPR()
            tpr, cg_idx, ring_idx = torch.tensor(tpr).float() / 10.0, \
                                    torch.tensor(cg_idx).to(torch.long), \
                                    torch.tensor(ring_idx).to(torch.long)

            rot, t = find_rigid_alignment(tpr[ring_idx], pos[mask])
            tpr = (rot.mm(tpr[ring_idx].T)).T + t

            shift = pos[mask_CG] - tpr[cg_idx]
            pos[mask & (atoms_type == amber_ff_atomstype.index('CZ'))] = torch.stack(
                ((tpr[ring_idx[0]] + shift).squeeze(0),
                 (tpr[ring_idx[2]] + shift).squeeze(0)), dim=0)
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
            pass
    mol = md.load_pdb(pdb)
    mol.xyz = np.append(mol.xyz, [pos.cpu().numpy()], axis=0)
    mol.save_pdb('generated_pdb/test_TPR_fix.pdb')


def test_vamp():
    import tensorflow as tf
    EPS = 1e-15

    def vampe_loss(X0, X1):
        b = tf.cast((tf.shape(X0)[0]), tf.float32)
        c00 = 1 / b * (tf.transpose(X0) @ X0)
        c11 = 1 / b * (tf.transpose(X1) @ X1)
        c01 = 1 / b * (tf.transpose(X0) @ X1)
        gamma_dia_inv = tf.linalg.diag(
            1 / (tf.reduce_mean(X1, axis=0) + EPS))  # add something so no devide by zero\n",
        first_term = c00 @ gamma_dia_inv @ c11 @ gamma_dia_inv
        second_term = 2 * (c01 @ gamma_dia_inv)
        vampe_arg = first_term - second_term
        vampe = tf.linalg.trace(vampe_arg)
        return vampe

    def vampe_loss_torch(X0: torch.Tensor, X1: torch.Tensor):
        b = float(X0.shape[0])
        c00 = 1 / b * (X0.T @ X0)
        c11 = 1 / b * (X1.T @ X1)
        c01 = 1 / b * (X0.T @ X1)
        gamma_dia_inv = torch.diag(1 / (torch.mean(X1, dim=0) + EPS))  # add something so no devide by zero\n",
        first_term = c00 @ gamma_dia_inv @ c11 @ gamma_dia_inv
        second_term = 2 * (c01 @ gamma_dia_inv)
        vampe_arg = first_term - second_term
        vampe = torch.trace(vampe_arg)
        return vampe

    a = torch.softmax(torch.randn((10, 5)), dim=-1).numpy()
    b = torch.softmax(torch.randn((10, 5)), dim=-1).numpy()
    e = vampe_loss(tf.convert_to_tensor(a), tf.convert_to_tensor(b))
    e1 = vampe_loss_torch(torch.tensor(a), torch.tensor(b))
    print(e, e1, e - e1)
    return True


from simtk.unit import *

nanoseconds = nano * seconds


def run():
    from molecule import Molecule
    import os
    import utils.utility as ut
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    mol = Molecule('5awl', '/home/giacomo/Documents/DeepMD/train_MoDEL/train_cing/5awl', False)
    mol.protein = ut.Protein(protein_name=mol.pdb_name, protein=mol.pdb_file, working_dir=mol.folder)
    mol.run(100 * nanoseconds, 20, False)


def remove_h():
    import mdtraj as md
    for i in range(6):
        trj = md.load_dcd(f'/home/giacomo/Documents/DeepMD/train_MoDEL/train_folding/1l2y/1l2y{i}.dcd',
                          top='/home/giacomo/Documents/DeepMD/train_MoDEL/train_folding/1l2y/1l2y.pdb')
        h_atoms = trj.trj.select('mass > 1.5')
        trj.atom_slice(h_atoms).save_dcd(
            f'/home/giacomo/Documents/DeepMD/train_MoDEL/train_folding/1l2y/1l2y{i}_noh.dcd')


def roba():
    from numpy import genfromtxt
    my_data = genfromtxt('TEK00003_30cm.CSV', delimiter=',')
    my_data = my_data[np.logical_not(np.isnan(my_data[:, 0])), :]
    from scipy.stats import norm
    (mu, sigma) = norm.fit(- 0.0004857 * my_data[:, 1] - 3.026e-05)
    print(mu, sigma)
    my_data = genfromtxt('TEK00004_60cm.CSV', delimiter=',')
    my_data = my_data[np.logical_not(np.isnan(my_data[:, 0])), :]
    from scipy.stats import norm
    (mu, sigma) = norm.fit(- 0.0004857 * my_data[:, 1] - 3.026e-05)
    print(mu, sigma)


def run_mcmch():
    from MCMCHybrid import MCMCHybridMove, WeightedMove
    from Markov import EdGE, noise_dim, latent_dim
    from openmmtools.mcmc import GHMCMove, HMCMove, IntegratorMove, MCDisplacementMove, MCRotationMove
    from utils.ModelAnalysis import ModelAnalysis
    from utils.dataset import DeepMDDataset
    from simtk import unit

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
    batch_size = database.trj_frames

    pg, _ = next(database)

    model = EdGE(input_features=35, noise_dim=noise_dim, ckpt_grad=False, latent_dim=latent_dim,
                 hiddens_en=[60, 90, 60, 40, 15],
                 hiddens_ha=[60, 90, 120, 90, 60, 50, 30, 20, 3])
    batch_size = database.trj_frames
    file_ck = f'models/ckpt_model_{batch_size}_{database.lag}_cing_M_4_CK.pt'
    file_train_ck = f'models/ckpt_model_{batch_size}_{database.lag}_cing_M_BT_4_CK.pt'
    file = f'models/ckpt_model_{batch_size}_{database.lag}_cing_M_4.pt'

    checkpoint_net = torch.load(file_train_ck)
    model.load_state_dict(checkpoint_net['model_state_dict'])
    model.to(model.device)

    probs = [0.1, 0.2, 0.3, 0.4]
    # n_frames = [1000]
    n_frames = []
    n_frames.extend([150000])
    n_frames.extend([1500 for p in probs])
    # n_iter = [10]
    n_iter = []
    n_iter.extend([1])
    n_iter.extend([1 for p in probs])
    md = ModelAnalysis()
    pdb_name = '5awl'
    md.move_comparison(pg, database, model, pdb_name, n_frames=n_frames, out_png='test/5awl', lag=5, probs=probs)


def autocorr():
    from utils.ModelAnalysis import ModelAnalysis
    from utils.dataset import DeepMDDataset
    md = ModelAnalysis()
    probs = [0.40, 0.60, 0.80, 0.90, 0.95, 1.00]
    labels = []
    labels.extend(['pivot'])
    labels.extend([f'{p}' for p in probs])
    md.move_autocorrelation(labels, 20)

# run()
run_mcmch()
#autocorr()
