import os
import os.path as osp
from typing import List

import mdtraj as md
import numpy as np
import torch
import torch.nn.functional as func
from simtk.unit import *
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data
from torch_geometric.data.dataloader import Collater
from prody import parsePDB

from molecule import Molecule
from utils.ProteinGraph import ProteinGraph

nanosecond = nano * second


class AutoMDS(torch.nn.Module):
    def __init__(self, n_atoms, max_iter, epsilon):
        super(AutoMDS, self).__init__()
        MAX_ITER = 100000
        self.max_iter = max_iter if max_iter > 0 else MAX_ITER
        self.pos = torch.nn.Parameter(torch.randn((n_atoms, 3)), requires_grad=True)
        self.pos.requires_grad = True
        self.epsilon = epsilon

    def fit(self, d_ij: torch.Tensor):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        assert d_ij.shape[0] == self.pos.shape[0]
        for i in range(self.max_iter):
            optimizer.zero_grad()
            d = torch.cdist(self.pos, self.pos)
            loss = func.mse_loss(d, d_ij)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print(loss.item())
                if loss.item() < self.epsilon:
                    break

    def reset(self):
        self.pos.data = torch.randn_like(self.pos)

    def fit_edges(self, edges_indx, edges_length):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        # assert edges_indx.max() == self.pos.shape[0]
        row, col = edges_indx
        for i in range(self.max_iter + 1):
            optimizer.zero_grad()
            d = torch.sqrt(torch.sum(torch.pow(self.pos[row] - self.pos[col], 2), dim=1)).requires_grad_(True)
            loss = func.mse_loss(d, edges_length)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print(f' iter: {i:>5d} fit mse: {loss.item():>5.3f}')
                if loss.item() < self.epsilon:
                    break


class DeepMDDataset:
    def __init__(self,
                 root=None,
                 topologies_files: List[str] = None,
                 trajectories_files: dict = None,
                 lag: int = None,
                 save_memory_mode: bool = True,
                 trj_frames: int = None,
                 log_file: str = None):

        """ If you want to generate a trajectory you must set get_trj=True and trj_frames will be the number of edges in
            your trajectory, while num_of_frames is the number of frame in trj_files"""
        if trajectories_files is None:
            trajectories_files = {}
        self.top_files = topologies_files
        self.trj_files = trajectories_files
        self.pt_files = {}
        self.num_frames = {}
        self.root = root
        self.lag = lag
        self.save_memory = save_memory_mode
        self.trj_frames = trj_frames
        self.is_train = True
        self.log_file = log_file
        self.max_atoms = 1000
        self.max_frame = None
        self.last_file_id = 0
        if self.log_file is not None:
            with open(self.log_file, 'w') as file:
                file.write('')

    def train(self):
        self.is_train = True

    def test(self):
        self.is_train = False

    def reinitialize(self):
        self.num_frames = {}
        for k, top_file in enumerate(self.top_files):
            name = top_file.split('/')[-1]
            name = name if '.pdb' not in name else name[:-4]
            self.num_frames[name] = 0.0
            wr_dir = osp.join(self.root, name)
            self.pt_files[name] = {'edges_index': osp.join(wr_dir, 'tensors', 'edges_index_{}.pt'.format(name)),
                                   'x': osp.join(wr_dir, 'tensors', 'x_{}.pt'.format(name)),
                                   'edges_attr': osp.join(wr_dir, 'tensors', 'edges_attr_{}.pt'.format(name)),
                                   'edges_length': []}
            for i in range(len(self.trj_files[name])):
                self.pt_files[name]['edges_length'].append(
                    osp.join(wr_dir, 'tensors', 'edges_length_{}_{}.pt'.format(name, i)))
                if self.num_frames[name] == 0:
                    self.num_frames[name] = torch.load(self.pt_files[name]['edges_length'][i]).shape[0]
                self.num_frames[name] = np.min([torch.load(self.pt_files[name]['edges_length'][i]).shape[0],
                                                self.num_frames[name]])

    @staticmethod
    def __process_traj_file(pt_file, edges_idx, traj_file, mol: Molecule, name, root, save_distance_matrix=False):
        mol.load_trajectory(traj_file)
        mol.save_trajectory_edges_length(root, name, pt_file, edges_idx, save_distance_matrix=save_distance_matrix)

    @staticmethod
    def process_pdb_dcd_file(pdb_file, dcd_file, root, name):
        if not osp.exists(osp.join(root, 'tensors')):
            os.mkdir(osp.join(root, 'tensors'))
        mol = Molecule(pdb_file, root)
        mol.pdb_noh, _ = mol.remove_from_pdb()
        mol.load_trajectory(dcd_file)
        proteingraph = ProteinGraph(mol.pdb_name, mol)
        proteingraph.save(root)
        mol.pdb_noh, _ = mol.remove_from_pdb()
        print('Processed file {}'.format(pdb_file))

    @property
    def raw_file_names(self):
        return self.top_files, self.trj_files

    @property
    def processed_file_names(self):
        return self.pt_files

    def get_n_pdb(self) -> int:
        return len(self.pt_files.keys())

    def get_pdb_names(self) -> list:
        return list(self.pt_files.keys())

    def len(self) -> int:
        return len(self.processed_file_names)

    def __report(self, pdb_name, file_id, from_i, to_j):
        if self.log_file is not None:
            with open(self.log_file, 'a') as file:
                file.write(
                    f'loaded: {pdb_name:>10} file_id: {file_id:>3} frames: {from_i:>5} to: {to_j:>5} lag: {self.lag:>3}\n')

    def load(self, idx_time, pdb_name=None, file_id=None, get_data=True):
        edge_attr = torch.load(self.pt_files[pdb_name]['edges_attr'])
        edge_length = torch.load(self.pt_files[pdb_name]['edges_length'][file_id])[idx_time]
        edge_index = torch.load(self.pt_files[pdb_name]['edges_index']).t()
        length_pos = 0
        edge_attr[:, length_pos] = edge_length[:]
        if get_data:
            x = torch.load(self.pt_files[pdb_name]['x'])
            return Data(x=x, edge_index=edge_index.t(), edge_attr=edge_attr)
        else:
            return edge_attr

    def __load_trj_frames(self, trj: md.Trajectory, idx_time, pdb_name=None):
        edge_attr = torch.load(self.pt_files[pdb_name]['edges_attr'])
        edge_index = torch.load(self.pt_files[pdb_name]['edges_index']).t()
        length_pos = 0
        x = torch.load(self.pt_files[pdb_name]['x'])
        d_ij = torch.tensor(pairwise_distances(X=trj.xyz[idx_time], metric='euclidean', n_jobs=-1))
        edge_attr[:, length_pos] = d_ij[edge_index[:, 0], edge_index[:, 1]]
        return Data(x=x, edge_index=edge_index.t(), edge_attr=edge_attr), d_ij

    def process(self, root_database):
        folders = []
        for subdir, dirs, files in os.walk(root_database):
            name = subdir.split(os.sep)[-1]
            if os.path.exists(os.path.join(subdir, name + '.pdb')) and (
                    os.path.exists(os.path.join(subdir, name + '.dcd')) or
                    os.path.exists(os.path.join(subdir, name + '0.dcd'))):
                folders.append([name, subdir, os.path.join(subdir, name + '.dcd')
                if os.path.exists(os.path.join(subdir, name + '.dcd')) else os.path.join(subdir, name + '0.dcd'),
                                os.path.join(subdir, name + '.pdb')])

        for folder in folders:
            # try:
            self.process_pdb_dcd_file(folder[3], folder[2], folder[1], folder[0])
            # except:
            #    pass

    def load_folder(self, root_database):
        folders = []
        self.root = root_database
        self.num_frames = {}
        self.trj_files = {}
        for subdir, dirs, files in os.walk(root_database):
            name = subdir.split(os.sep)[-1]
            name = name
            if os.path.exists(os.path.join(subdir, name + '.pdb')) and \
                    (os.path.exists(os.path.join(subdir, name + '.dcd')) or os.path.exists(
                        os.path.join(subdir, name + '0.dcd'))):
                trajectories = []
                i = 0
                if os.path.exists(os.path.join(subdir, name + '.dcd')):
                    trajectories.append(os.path.join(subdir, name + '.dcd'))
                while os.path.exists(os.path.join(subdir, name + '{}.dcd'.format(i))):
                    trajectories.append(os.path.join(subdir, name + '{}.dcd'.format(i)))
                    i += 1
                folders.append([name, subdir, trajectories, os.path.join(subdir, name + '.pdb')])
        for folder in folders:
            name = folder[0]
            subdir = folder[1]
            self.num_frames[name] = 0.0
            self.pt_files[name] = {'edges_index': osp.join(subdir, 'tensors', 'edges_index_{}.pt'.format(name)),
                                   'x': osp.join(subdir, 'tensors', 'x_{}.pt'.format(name)),
                                   'edges_attr': osp.join(subdir, 'tensors', 'edges_attr_{}.pt'.format(name)),
                                   'edges_length': []}
            self.trj_files[name] = folder[2]
            i = 0
            if os.path.exists(os.path.join(subdir, 'tensors', 'edges_length_{}_{}.pt'.format(name, i))):
                self.pt_files[name]['edges_length'].append(
                    osp.join(subdir, 'tensors', 'edges_length_{}_{}.pt'.format(name, i)))

            i += 1
            while os.path.exists(os.path.join(subdir, 'tensors', 'edges_length_{}_{}.pt'.format(name, i))):
                self.pt_files[name]['edges_length'].append(
                    osp.join(subdir, 'tensors', 'edges_length_{}_{}.pt'.format(name, i)))
                i += 1

    def __iter__(self):
        return self

    def __next__(self):
        return self.__get_random_trj_save_memory()

    def __getitem__(self, item):
        return self.__get_random_trj_save_memory()

    def print_n_atoms(self, n_max=None):
        list_atoms = []
        for pdb in self.pt_files.keys():
            n_atoms = parsePDB(os.path.join(self.root, pdb, pdb + '.pdb')).numAtoms()
            if n_max is not None:
                if n_atoms < n_max:
                    list_atoms.append([pdb, n_atoms])
            else:
                print(f' {pdb:>4s} {n_atoms:>5d}')
        return list_atoms

    def get_trajectory(self, pdb_name, file_id, frame_i=None, frame_j=None, lag=None):
        if lag is None:
            lag = self.lag
            pass
        try:
            pdb_file = os.path.join(self.root, pdb_name, pdb_name + '.pdb')
            dcd_file = self.trj_files[pdb_name][file_id]
            trj: md.Trajectory = md.load_dcd(dcd_file, pdb_file)
        except:
            pdb_file = os.path.join(self.root, pdb_name, pdb_name + '_noh.pdb')
            dcd_file = self.trj_files[pdb_name][file_id]
            trj: md.Trajectory = md.load_dcd(dcd_file, pdb_file)

        num_frames = trj.n_frames
        if frame_i is None or frame_j is None:
            assert num_frames >= int(self.lag * self.trj_frames)
            frame_i = np.random.randint(0, int(num_frames - self.lag * self.trj_frames))
            frame_j = frame_i + self.trj_frames * self.lag
        elif frame_j < 0 and frame_i == 0:
            frame_j = num_frames - 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_trj = []
        root = os.path.join(self.root, pdb_name)
        pg = ProteinGraph(pdb_name)
        pg.load(root)

        for i in range(frame_i, frame_j, lag):
            pos = torch.tensor(trj.xyz[i], device=device)
            # edges_length = torch.sqrt(torch.sum(torch.pow(pos[row] - pos[col], 2), dim=1))
            tensor_trj.append(pos)
        return pg, torch.stack(tensor_trj)

    def __get_random_trj_save_memory(self):
        """ Return a trajectory of trj_frames with lag time tau. The pdb is chosen randomly"""
        while True:
            pdb_name = self.get_pdb_names()[np.random.randint(0, self.get_n_pdb())]
            file_id = self.last_file_id + 1  # np.random.randint(0, len(self.trj_files[pdb_name]))
            if file_id >= len(self.trj_files[pdb_name]):
                file_id = 0
            self.last_file_id = file_id

            try:
                pdb_file = os.path.join(self.root, pdb_name, pdb_name + '_noh.pdb')
                dcd_file = self.trj_files[pdb_name][file_id]
                trj: md.Trajectory = md.load_dcd(dcd_file, pdb_file)
            except:
                pdb_file = os.path.join(self.root, pdb_name, pdb_name + '.pdb')
                dcd_file = self.trj_files[pdb_name][file_id]
                trj: md.Trajectory = md.load_dcd(dcd_file, pdb_file)
                trj = trj.atom_slice(trj.top.select('mass > 1.5'))

            num_frames = trj.n_frames
            assert num_frames >= int(self.lag * self.trj_frames)
            if self.max_frame is None:
                frame_i = np.random.randint(2, int(num_frames - self.lag * self.trj_frames))
                frame_j = frame_i + self.trj_frames * self.lag
            else:
                assert self.max_frame <= num_frames
                assert self.max_frame >= self.trj_frames * self.lag
                frame_j = np.random.randint(self.trj_frames * self.lag, self.max_frame)
                frame_i = int(max(frame_j - self.trj_frames * self.lag, 2))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            root = os.path.join(self.root, pdb_name)
            pg = ProteinGraph(pdb_name)
            pg.load(root)
            if pg.x.shape[0] > self.max_atoms:
                continue

            row, col = pg.edge_index

            tensor_trj = []
            for i in range(frame_i, frame_j, self.lag):
                pos = torch.tensor(trj.xyz[i], device=device)
                # edges_length = torch.sqrt(torch.sum(torch.pow(pos[row] - pos[col], 2), dim=1))
                tensor_trj.append(pos)
                if i == frame_i:
                    pg.edge_attr[:, 0] = torch.sqrt(torch.sum(torch.pow(pos[row] - pos[col], 2), dim=1))

            self.__report(pdb_name, file_id, frame_i, frame_j)
            tensor_trj = torch.stack(tensor_trj)
            tensor_trj.requires_grad_(True)
            pg.edge_attr.requires_grad_(True)
            pg.x.requires_grad_(True)
            return pg, tensor_trj

    @staticmethod
    def reduce_edges_length(edges_length: torch.Tensor):
        idx1 = torch.arange(0, edges_length.shape[0], 2, device=edges_length.device, dtype=torch.long)
        idx2 = idx1 + torch.ones_like(idx1, device=edges_length.device, dtype=torch.long)
        return (edges_length[idx1] + edges_length[idx2]) / 2.0


class DeepMDDataLoader:
    def __init__(self, dataset: DeepMDDataset, batch_size=10, follow_batch=None, test_set: DeepMDDataset = None):
        if follow_batch is None:
            follow_batch = []
        self.dataset = dataset
        self.dataset.train()
        self.batch_size = batch_size
        self.collate = Collater(follow_batch=follow_batch)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_train = False
        self.test_set = test_set

    def train(self):
        self.dataset.train()
        self.is_train = True

    def test(self):
        self.dataset.test()
        self.is_train = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size > 1:
            data_list = []
            trj_list = []
            pdb_names = []
            for _ in range(self.batch_size):
                if (not self.is_train) and (self.test_set is not None):
                    data, trj, pdb_name = next(self.test_set)
                else:
                    data, trj, pdb_name = next(self.dataset)
                data_list.append(data)
                trj_list.append(trj)
                pdb_names.append(pdb_name)
            batch = self.collate(data_list)
            batch_trj = torch.stack(trj_list)
            return batch.to(self.device), batch_trj
        else:
            if (not self.is_train) and (self.test_set is not None):
                data, trj = next(self.test_set)
            else:
                data, trj = next(self.dataset)
            return data, trj
        # batch_trj shape: (n_batches, n_frames, n_edges, 1)
