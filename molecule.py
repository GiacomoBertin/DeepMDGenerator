from rdkit import Chem
import torch
from torch_geometric.data import Data
from prody import *
from utils import utility as ut
import mdtraj as md
from getcontacts.get_static_contacts import main
from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
import pyemma
import matplotlib.pyplot as plt
from utils.utility import FileUtility
from sklearn.neighbors import NearestNeighbors
import numpy as np

RAD2DEG = 180 / np.pi
picosecond = pico * second
nanometer = nano * meter
nanosecond = nano * second


def onek_encoding_unk(x, allowable_set):
    """
    Function for one hot encoding
    :param x: value to one-hot
    :param allowable_set: set of options to encode
    :return: one-hot encoding as torch tensor
    """
    # if x not in allowable_set:
    #    x = allowable_set[-1]
    return [x == s for s in allowable_set]


def compute_distance(coords1, coords2):
    """Returns the distance"""

    a1 = coords2 - coords1
    return np.sqrt(np.dot(a1, a1))


def get_structure(atom_name):
    if atom_name.replace(' ', '') in ['CA', 'CB', 'O', 'N', 'C']:
        return 'bb'
    else:
        return 'sc'


def compute_inversion_angle(coordsn, coordsk, coordsi, coordsj, radian=False):
    """Returns the dihedral angle in degrees unless ``radian=True``."""

    rik = coordsk - coordsi
    rin = coordsn - coordsi
    rij = coordsj - coordsi
    ukn = (rik + rin) / np.sqrt(np.dot(rik + rin, rik + rin))
    vkn = (rik - rin) / np.sqrt(np.dot(rik - rin, rik - rin))

    rad = np.arccos(np.sqrt(np.dot(rij, vkn) ** 2 + np.dot(rij, ukn) ** 2) / np.sqrt(np.dot(rij, rij)))

    if radian:
        return rad
    else:
        return rad * RAD2DEG


def compute_dihedral(coords1, coords2, coords3, coords4, radian=False):
    """Returns the dihedral angle in degrees unless ``radian=True``."""

    a1 = coords2 - coords1
    a2 = coords3 - coords2
    a3 = coords4 - coords3

    v1 = np.cross(a1, a2)
    v1 = v1 / (v1 * v1).sum(-1) ** 0.5
    v2 = np.cross(a2, a3)
    v2 = v2 / (v2 * v2).sum(-1) ** 0.5
    porm = np.sign((v1 * a3).sum(-1))
    rad = np.arccos((v1 * v2).sum(-1) / ((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1)) ** 0.5)
    if not porm == 0:
        rad = rad * porm
    if radian:
        return rad
    else:
        return rad * RAD2DEG


def compute_angle(coords1, coords2, coords3, radian=False):
    """Returns bond angle in degrees unless ``radian=True``"""

    v1 = coords1 - coords2
    v2 = coords3 - coords2

    rad = np.arccos((v1 * v2).sum(-1) / ((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1)) ** 0.5)
    if radian:
        return rad
    else:
        return rad * RAD2DEG


residues_encoded = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                    'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

amber_ff = """C                 12.01000	; sp2 C carbonyl group 
CA                12.01000	; sp2 C pure aromatic (benzene)
CB                12.01000	; sp2 aromatic C, 5&6 membered ring junction
CC                12.01000	; sp2 aromatic C, 5 memb. ring HIS
CK                12.01000	; sp2 C 5 memb.ring in purines
CM                12.01000	; sp2 C  pyrimidines in pos. 5 & 6
CN                12.01000	; sp2 C aromatic 5&6 memb.ring junct.(TRP)
CQ                12.01000	; sp2 C in 5 mem.ring of purines between 2 N
CR                12.01000	; sp2 arom as CQ but in HIS
CT                12.01000	; sp3 aliphatic C
CV                12.01000	; sp2 arom. 5 memb.ring w/1 N and 1 H (HIS)
CW                12.01000	; sp2 arom. 5 memb.ring w/1 N-H and 1 H (HIS)
CG                12.01000	; sp2 arom. 5 memb.ring w/1 subst. (TRP)
CD                12.01000	; sp2 arom. 5 memb.ring w/1 subst. (TRP)
CE                12.01000	; sp2 arom. 5 memb.ring w/1 subst. (TRP)
CZ                12.01000	; sp2 arom. 5 memb.ring w/1 subst. (TRP)
CH                12.01000	; sp2 arom. 5 memb.ring w/1 subst. (TRP)
C*                12.01000	; sp2 arom. 5 memb.ring w/1 subst. (TRP)
C0                40.08000	; calcium
F                 19.00000	; fluorine
H                  1.00800	; H bonded to nitrogen atoms
HC                 1.00800	; H aliph. bond. to C without electrwd.group
H1                 1.00800	; H aliph. bond. to C with 1 electrwd. group
H2                 1.00800	; H aliph. bond. to C with 2 electrwd.groups
H3                 1.00800	; H aliph. bond. to C with 3 eletrwd.groups
HA                 1.00800	; H arom. bond. to C without elctrwd. groups
H4                 1.00800	; H arom. bond. to C with 1 electrwd. group
H5                 1.00800	; H arom. bond. to C with 2 electrwd. groups
HO                 1.00800	; hydroxyl group
HS                 1.00800	; hydrogen bonded to sulphur (pol?)
HW                 1.00800	; H in TIP3P water
HP                 1.00800	; H bonded to C next to positively charged gr
N                 14.01000	; sp2 nitrogen in amide groups
NA                14.01000	; sp2 N in 5 memb.ring w/H atom (HIS)
NB                14.01000	; sp2 N in 5 memb.ring w/LP (HIS,ADE,GUA)
NC                14.01000	; sp2 N in 6 memb.ring w/LP (ADE,GUA)
N2                14.01000	; sp2 N in amino groups
N3                14.01000	; sp3 N for charged amino groups (Lys, etc)
NE                14.01000	; sp2 N 
N*                14.01000	; sp2 N 
O                 16.00000	; carbonyl group oxygen
OX                16.00000	; carbonyl group oxygen
OW                16.00000	; oxygen in TIP3P water
OH                16.00000	; oxygen in hydroxyl group
O*                16.00000	; oxygen
OS                16.00000	; ether and ester oxygen
O2                16.00000	; carboxyl and phosphate group oxygen
P                 30.97000	; phosphate,pol:JACS,112,8543,90,K.J.Miller
S                 32.06000	; S in disulfide linkage,pol:JPC,102,2399,98
SH                32.06000	; S in cystine"""

amber_ff_atomstype = []
for line in amber_ff.split('\n'):
    amber_ff_atomstype.append(line.split()[0])


class Atom:
    def __init__(self,
                 name,
                 idx: int = None,
                 residue: str = None,
                 radii: float = None,
                 charge: float = None,
                 element: str = None,
                 res_idx: int = None,
                 is_in_ring: bool = False,
                 is_aromatic: bool = False,
                 ):
        self.name = name.replace(' ', '')
        self.idx = idx
        self.residue = residue
        self.radii = radii
        self.charge = charge
        self.structure = 'bb' if name in ['O', 'C', 'N', 'CA'] else 'ss'
        self.is_bb = self.structure == 'bb'
        self.is_calpha = self.name == 'CA'
        self.element = element
        self.res_idx = res_idx
        self.is_in_ring = is_in_ring
        self.is_aromatic = is_aromatic
        self.atom_group = None
        self.atom_type = name.replace(' ', '')[:2]
        if 'C' in self.atom_type:
            if self.atom_type not in amber_ff_atomstype:
                self.atom_type = 'C*'
        elif 'O' in self.atom_type:
            if self.atom_type not in amber_ff_atomstype:
                self.atom_type = 'O*'
        elif 'N' in self.atom_type:
            if self.atom_type not in amber_ff_atomstype:
                self.atom_type = 'N*'

        if self.name.startswith('H'):
            self.atom_group = (self.get_hydrogen_code(), self.get_all_atoms_code())
        elif self.name in ['O', 'C', 'N']:
            self.atom_group = (self.get_backbones_atoms_code(), self.get_all_atoms_code(), self.get_heavy_atoms_code())
        elif self.name in ['CA']:
            self.atom_group = (self.get_calpha_atoms_code(), self.get_backbones_atoms_code(), self.get_all_atoms_code(),
                               self.get_heavy_atoms_code())
        else:
            self.atom_group = (self.get_heavy_atoms_code(), self.get_all_atoms_code())

    def to_dict(self):
        return {
            'name': self.name,
            'idx': self.idx,
            'residue': self.residue,
            'radii': self.radii,
            'charge': self.charge,
            'structure': self.structure,
            'is_bb': self.is_bb,
            'is_calpha': self.is_calpha,
            'element': self.element,
            'res_idx': self.res_idx,
            'is_in_ring': self.is_in_ring,
            'is_aromatic': self.is_aromatic,
            'atom_group': self.atom_group,
            'atom_type': self.atom_type
        }

    @staticmethod
    def get_all_atoms_code() -> str:
        return 'aa'

    @staticmethod
    def get_hydrogen_code() -> str:
        return 'h'

    @staticmethod
    def get_heavy_atoms_code() -> str:
        return 'ha'

    @staticmethod
    def get_backbones_atoms_code() -> str:
        return 'bb'

    @staticmethod
    def get_calpha_atoms_code() -> str:
        return 'ca'

    def __repr__(self):
        return f"{self.idx:>5} {self.name:>4} {self.element:>2} {self.res_idx:>4} {self.residue:>4} {self.charge:>8} {self.radii:>8}\n"

    def get_atom_groups(self) -> tuple:
        return self.atom_group


class Edge:
    def __init__(self, begin_atom: int, end_atom: int, length: float = 0.0, edge_type: str = None,
                 features: dict = None):
        self.begin_atom = begin_atom
        self.end_atom = end_atom
        self.length = length
        self.type = edge_type
        self.features = features
        self.bond = None
        self.dihedral_i = None
        self.dihedral_j = None
        self.angles_i = None
        self.angles_j = None
        self.is_in_ring = False
        self.is_aromatic = False
        self.is_covalent_bond = False
        self.connect_calpha = False

    def chem_bond_info(self, mybond: Chem.Bond):
        self.bond = mybond

    def compute_angles(self, positions):
        begin_atom = self.bond.GetBeginAtom()
        end_atom = self.bond.GetEndAtom()
        bonds_begin = list(begin_atom.GetBonds())
        bonds_end = list(end_atom.GetBonds())
        bond = self.bond
        for i in range(len(bonds_begin)):
            if bonds_begin[i].GetBeginAtomIdx() != bond.GetBeginAtomIdx():
                atom_i = bonds_begin[i].GetBeginAtom()
            else:
                atom_i = bonds_begin[i].GetEndAtom()
            angle_i = compute_angle(positions[atom_i.GetIdx()], positions[begin_atom.GetIdx()])
            self.angles_i.append(angle_i)

        for i in range(len(bonds_end)):
            if bonds_end[i].GetBeginAtomIdx() != bond.GetEndAtomIdx():
                atom_i = bonds_end[i].GetBeginAtom()
            else:
                atom_i = bonds_end[i].GetEndAtom()
            angle_i = compute_angle(positions[end_atom.GetIdx()], positions[atom_i.GetIdx()])
            self.angles_j.append(angle_i)

    def compute_inversion_angles(self, positions):
        begin_atom = self.bond.GetBeginAtom()
        end_atom = self.bond.GetEndAtom()
        bonds_begin = list(begin_atom.GetBonds())
        bonds_end = list(end_atom.GetBonds())
        bond = self.bond
        for i in range(len(bonds_begin) - 1):
            if bonds_begin[i].GetBeginAtomIdx() != bond.GetBeginAtomIdx():
                atom_i = bonds_begin[i].GetBeginAtom()
            else:
                atom_i = bonds_begin[i].GetEndAtom()

            for j in range(i + 1, len(bonds_begin)):
                if bonds_begin[j].GetBeginAtomIdx() != bond.GetBeginAtomIdx():
                    atom_j = bonds_begin[j].GetBeginAtom()
                else:
                    atom_j = bonds_begin[j].GetEndAtom()

                dihedral_ij = compute_inversion_angle(positions[atom_i.GetIdx()],
                                                      positions[atom_j.GetIdx()],
                                                      positions[begin_atom.GetIdx()],
                                                      positions[end_atom.GetIdx()])
                self.dihedral_i.append(dihedral_ij)

        for i in range(len(bonds_end) - 1):
            if bonds_end[i].GetBeginAtomIdx() != bond.GetEndAtomIdx():
                atom_i = bonds_end[i].GetBeginAtom()
            else:
                atom_i = bonds_end[i].GetEndAtom()

            for j in range(i + 1, len(bonds_end)):
                if bonds_end[j].GetBeginAtomIdx() != bond.GetEndAtomIdx():
                    atom_j = bonds_end[j].GetBeginAtom()
                else:
                    atom_j = bonds_end[j].GetEndAtom()

                dihedral_ij = compute_inversion_angle(positions[atom_i.GetIdx()],
                                                      positions[atom_j.GetIdx()],
                                                      positions[end_atom.GetIdx()],
                                                      positions[begin_atom.GetIdx()])
                self.dihedral_j.append(dihedral_ij)

    def __eq__(self, other):
        if isinstance(other, Edge):
            return ((self.begin_atom == other.begin_atom) and (self.end_atom == other.end_atom)) or \
                   ((self.end_atom == other.begin_atom) and (self.begin_atom == other.end_atom))
        elif isinstance(other, Chem.Bond):
            return ((self.begin_atom == other.GetBeginAtom().GetIdx()) and (
                    self.end_atom == other.GetEndAtom().GetIdx())) or \
                   ((self.end_atom == other.GetBeginAtom().GetIdx()) and (
                           self.begin_atom == other.GetEndAtom().GetIdx()))
        else:
            return ((self.begin_atom == other[0]) and (self.end_atom == other[1])) or \
                   ((self.end_atom == other[0]) and (self.begin_atom == other[1]))

    def __repr__(self):
        return "{:5>d} {:5>d} {} {}\n".format(self.begin_atom, self.end_atom, self.type, self.features)


class Molecule:
    def __init__(self, pdb_name: str, folder: str, initialize: bool = True):

        # Define Features Available
        self.atoms_encoded = ['C', 'N', 'O', 'P', 'S', 'H']
        self.atoms_ff = amber_ff_atomstype
        self.atoms_structure_encoded = ['bb', 'sc']
        self.edge_encoded = ['nb', 'vdw', 'hp', 'ts', 'ps', 'pc', 'sb', 'hbbb', 'hbsb', 'hbss']
        self.edge_encoded.extend(['SINGLE', 'DOUBLE', 'AROMATIC', 'ZERO'])
        self.residues_encoded = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                                 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

        self.pdb_name = pdb_name.split('/')[-1] if ".pdb" not in pdb_name else pdb_name.split('/')[-1][:-4]
        self.folder = folder
        self.pdb_file = os.path.join(self.folder, self.pdb_name + ".pdb")
        self.mol2_file = None
        self.pdb_noh = None
        self.protein = None
        self.mol2_file = None
        self.rdkitMol = None
        self.edges_info = []
        self.atoms_info = []
        self.num_max_bonds = 0
        self.positions = None

        self.max_dihedrals = 0
        self.max_angles = 0

        self.max_edge_geometrical_features = 0
        # self.charges = []
        # self.radii = []
        self.trajectory = None
        self.top_file = None
        if initialize:
            self.initialize()
        self.angles = None
        self.dihedrals = None
        self.bonds = None

    def initialize(self):
        # NB: rdkitMol has always the hydrogen
        self.protein = ut.Protein(protein_name=self.pdb_name, protein=self.pdb_file, working_dir=self.folder)
        self.mol2_file = self.protein.save_as_mol2()
        # self.rdkitMol: Chem.Mol = Chem.MolFromPDBFile(self.pdb_file)
        # if self.rdkitMol is None:
        self.rdkitMol: Chem.Mol = Chem.MolFromMol2File(self.mol2_file, sanitize=False)

        self.edges_info = []
        self.atoms_info = []
        self.num_max_bonds = np.max([len(my_atom.GetBonds()) for my_atom in self.rdkitMol.GetAtoms()])
        self.positions = np.array([c.GetPositions() / 10.0 for c in self.rdkitMol.GetConformers()])[0]

        self.max_dihedrals = self.num_max_bonds * (self.num_max_bonds - 1) / 2
        self.max_angles = self.num_max_bonds

        self.max_edge_geometrical_features = self.max_dihedrals + self.max_angles + 1
        self.top_file, _ = ut.AmberUtility.amb2gro_top_gro(self.protein.prmtop_file, self.protein.prmcrd_file,
                                                           self.pdb_file[:-4] + '.top', self.pdb_file[:-4] + '.gro')
        # os.system(f'rm {self.protein.working_dir}/#*')

    def load_ff(self, file_name):
        with open(file_name, 'r') as file:
            for line in file:
                self.atoms_ff.append(line.split()[0])

    def generate(self, group: str = None, generate_edges_bond=False):
        self.__initialize_atoms(group=group)
        self.__initialize_edges(group=group)
        self.__generate_contacts()
        self.__initialize_angles_and_dihedrals(group=group, generate_edges_bond=generate_edges_bond)
        pass

    def __initialize_angles_and_dihedrals(self, group: str = None, generate_edges_bond=False):
        self.bonds, self.angles, self.dihedrals = ut.GROMACSUtility.read_top_file(self.top_file)
        if generate_edges_bond:
            for b in self.bonds:
                self.__add_edge(b[0], b[1], group=group)
            for a in self.angles:
                self.__add_edge(a[0], a[1], group=group)
                self.__add_edge(a[1], a[2], group=group)
                self.__add_edge(a[2], a[0], group=group)

            for d in self.dihedrals:
                self.__add_edge(d[0], d[1], group=group)  # ij
                self.__add_edge(d[1], d[3], group=group)  # jl
                self.__add_edge(d[0], d[3], group=group)  # il

                # self.__add_edge(d[0], d[1], group=group)  # ij
                self.__add_edge(d[1], d[2], group=group)  # jk
                self.__add_edge(d[0], d[2], group=group)  # ik

                self.__add_edge(d[2], d[1], group=group)  # kj
                # self.__add_edge(d[1], d[3], group=group)  # jl
                self.__add_edge(d[2], d[3], group=group)  # kl

        # ijkl need:              0123
        # 0123
        # [[ij] [jl] [il]] (ijl)  01 13 03
        # [[ij] [jk] [ik]] (ijk)  01 12 02
        # [[kj] [jl] [kl]] (kjl)  21 13 23 oufff... messy

    def __initialize_atoms(self, group: str = None):
        self.positions = []
        ut.APBSUtility.pdb2pqr(self.pdb_file, self.pdb_file[:-4] + ".pqr")
        radii = []
        charges = []
        prq_loaded = False
        try:
            with open(self.pdb_file[:-4] + ".pqr", "r") as file:
                for line in file:
                    if "ATOM" in line:
                        words = line.split()
                        charge = words[-2]
                        radius = words[-1]
                        charges.append(float(charge))
                        radii.append(float(radius))
            prq_loaded = True
        except:
            prq_loaded = False
        i = 0
        with open(self.mol2_file, 'rb') as file:
            q = file.read()
            lines = q.decode('latin-1').split('\n')
            read = False
            for line in lines:
                if line.startswith('@<TRIPOS>ATOM'):
                    read = True
                    continue
                if line.startswith('@<TRIPOS>BOND'):
                    read = False
                    break
                if read:
                    words = line.split()
                    #   0 1          2          3            4     5     6   7    8     -2   -1
                    #   1 N      35.125000   57.952000   78.949000 N     1 NGLU  ???  0.0017 ****
                    idx = int(words[0])
                    name = words[1]
                    self.positions.append([float(words[2]), float(words[3]), float(words[4])])
                    element = words[5]
                    res_idx = int(words[6])
                    res_name = words[7]
                    charge = float(words[-2])
                    radius = radii[i]  # TODO take radii from mol2
                    # assert charge == charges[i]
                    self.atoms_info.append(Atom(name, idx, res_name, radius, charge, element, res_idx))
                    i += 1
        if self.rdkitMol is not None:
            for i, atom in enumerate(self.rdkitMol.GetAtoms()):
                self.atoms_info[i].is_in_ring = atom.IsInRing()
                self.atoms_info[i].is_aromatic = atom.GetIsAromatic()
        self.positions = np.array(self.positions)

    def __initialize_edges(self,
                           compute_angles=False,
                           compute_inversion_angles=False,
                           group=None):
        for bond in self.rdkitMol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            self.__add_edge(begin_atom.GetIdx(), end_atom.GetIdx(), bond, compute_angles, compute_inversion_angles,
                            bond_type=bond.GetBondType(), group=group)

    def __fast_recompute_lenghts(self, edges_index, edges_attr,
                                 length_pos: int = 0):  # edges_index[i] = [i_begin, i_end]
        d_ij = torch.tensor(distance_matrix(self.positions, self.positions))
        edges_attr[:, length_pos] = d_ij[edges_index[:, 0], edges_index[:, 1]]

    def __recompute_lengths(self):
        d_ij = distance_matrix(self.positions, self.positions)
        for i in range(len(self.edges_info)):
            self.edges_info[i].length = d_ij[self.edges_info[i].begin_atom, self.edges_info[i].end_atom]

    def save_trajectory_edges_length(self, root, name, file_pt_name, edges_index=None, step=10,
                                     save_distance_matrix=False):
        file_pt_name = file_pt_name[:-3] if ".pt" in file_pt_name else file_pt_name
        # nb: mdtraj use nanometers, while pdb use angstrom
        if edges_index is None:
            edges_index = []
            for e in self.edges_info:
                edges_index.append([e.begin_atom, e.end_atom])
                edges_index.append([e.end_atom, e.begin_atom])

            edges_index = torch.tensor(edges_index)

            edges_length = torch.zeros((self.trajectory.n_frames, 2 * len(self.edges_info)))
            for i in range(self.trajectory.n_frames):
                d_ij = distance_matrix(self.trajectory.xyz[i], self.trajectory.xyz[i])
                edges_length[i, :] = d_ij[edges_index[:, 0], edges_index[:, 1]]
            torch.save(edges_length, file_pt_name + ".pt")
            return file_pt_name + ".pt"
        else:
            edges_length = torch.zeros((int(self.trajectory.n_frames / step), edges_index.shape[0]))
            dist_folder = os.path.join(root, 'tensors', 'distances')
            if save_distance_matrix:
                if not os.path.exists(dist_folder):
                    os.mkdir(dist_folder)
            for i in range(0, self.trajectory.n_frames, step):
                d_ij = torch.tensor(pairwise_distances(X=self.trajectory.xyz[i], metric='euclidean', n_jobs=-1))
                if save_distance_matrix:
                    torch.save(d_ij, os.path.join(dist_folder, name + '_dij_{}.pt'.format(int(i / step))))
                edges_length[int(i / step), :] = d_ij[edges_index[:, 0], edges_index[:, 1]]
                if i % 100 == 0:
                    print(i)
            torch.save(edges_length, file_pt_name + ".pt")
            return file_pt_name + ".pt"

    def atoms_slices(self, group):
        index_map = [None] * len(self.atoms_info)
        new_atoms_info = []
        for i, atom in enumerate(self.atoms_info):
            if group in atom.get_atom_groups():
                new_i = len(new_atoms_info)
                new_atoms_info.append(atom)
                index_map[i] = new_i

        new_edges = []
        for i, bond in enumerate(self.edges_info):
            if group in self.atoms_info[bond.begin_atom].get_atom_groups() and \
                    group in self.atoms_info[bond.end_atom].get_atom_groups():
                self.edges_info[i].begin_atom = index_map[bond.begin_atom]
                self.edges_info[i].end_atom = index_map[bond.end_atom]
                new_edges.append(self.edges_info[i])

        new_bonds = []
        for i, bond in enumerate(self.bonds):
            if group in self.atoms_info[int(bond[0])].get_atom_groups() and \
                    group in self.atoms_info[int(bond[1])].get_atom_groups():
                self.bonds[i][0] = index_map[int(bond[0])]
                self.bonds[i][1] = index_map[int(bond[1])]
                new_bonds.append(self.bonds[i])

        new_angles = []
        for i, angle in enumerate(self.angles):
            if group in self.atoms_info[int(angle[0])].get_atom_groups() and \
                    group in self.atoms_info[int(angle[1])].get_atom_groups() and \
                    group in self.atoms_info[int(angle[2])].get_atom_groups():
                self.angles[i][0] = index_map[int(angle[0])]
                self.angles[i][1] = index_map[int(angle[1])]
                self.angles[i][2] = index_map[int(angle[2])]
                new_angles.append(self.angles[i])

        new_dihedrals = []
        for i, dihedral in enumerate(self.dihedrals):
            if group in self.atoms_info[int(dihedral[0])].get_atom_groups() and \
                    group in self.atoms_info[int(dihedral[1])].get_atom_groups() and \
                    group in self.atoms_info[int(dihedral[2])].get_atom_groups() and \
                    group in self.atoms_info[int(dihedral[3])].get_atom_groups():
                self.dihedrals[i][0] = index_map[int(dihedral[0])]
                self.dihedrals[i][1] = index_map[int(dihedral[1])]
                self.dihedrals[i][2] = index_map[int(dihedral[2])]
                self.dihedrals[i][3] = index_map[int(dihedral[3])]
                new_dihedrals.append(self.dihedrals[i])

        self.edges_info = new_edges
        self.atoms_info = new_atoms_info
        self.bonds = np.array(new_bonds)
        self.angles = np.array(new_angles)
        self.dihedrals = np.array(new_dihedrals)
        return True

    def recompute_edge_attr(self,
                            compute_angles: bool = False,
                            compute_inversion_angles: bool = False):
        edge_attr = []
        for edge in self.edges_info:
            edge: Edge
            feat = self.__get_features_edge(edge, False, compute_angles, compute_inversion_angles)
            edge_attr.append(feat)
            feat = self.__get_features_edge(edge, True, compute_angles, compute_inversion_angles)
            edge_attr.append(feat)

        return torch.tensor(edge_attr, dtype=torch.float)

    def __add_edge(self,
                   i: int,
                   j: int,
                   ij_bond: Chem.Bond = None,
                   compute_angles: bool = False,
                   compute_inversion_angles: bool = False,
                   bond_type: str = None,
                   group: str = None):
        i = int(i)
        j = int(j)
        if group is not None:
            if (group not in self.atoms_info[i].get_atom_groups()) or (
                    group not in self.atoms_info[j].get_atom_groups()):
                return None
        e = Edge(i, j)
        if e not in self.edges_info:
            e.connect_calpha = (self.atoms_info[j].name == 'CA') and (self.atoms_info[i].name == 'CA')

            if ij_bond is not None:
                begin_atom = ij_bond.GetBeginAtom() if i == ij_bond.GetBeginAtom().GetIdx() else ij_bond.GetEndAtom()
                end_atom = ij_bond.GetEndAtom() if j == ij_bond.GetEndAtom().GetIdx() else ij_bond.GetBeginAtom()
                e.is_covalent_bond = True
                e.bond = ij_bond
                e.type = ij_bond.GetBondType().name
                e.is_aromatic = ij_bond.GetIsAromatic()
                e.is_in_ring = ij_bond.IsInRing()
            else:
                begin_atom = self.rdkitMol.GetAtomWithIdx(i)
                end_atom = self.rdkitMol.GetAtomWithIdx(j)
                if bond_type is not None:
                    e.type = bond_type
                    e.is_covalent_bond = False
                else:
                    e.type = "nb"
                    e.is_covalent_bond = False

            # if ij_bond is None:
            #    ij_bond: Chem.Bond = self.rdkitMol.GetBondBetweenAtoms(i, j)

            e.length = compute_distance(self.positions[i], self.positions[j])

            assert i == begin_atom.GetIdx()
            assert j == end_atom.GetIdx()

            if compute_angles:
                e.compute_angles(self.positions)

            if compute_inversion_angles:
                e.compute_inversion_angles(self.positions)

            e.features = {"i_symbol": begin_atom.GetSymbol(),
                          "j_symbol": end_atom.GetSymbol(),
                          "i_charge": self.atoms_info[i].charge,
                          "j_charge": self.atoms_info[j].charge,
                          "i_radii": self.atoms_info[i].radii,
                          "j_radii": self.atoms_info[j].radii}
            self.edges_info.append(e)
            return e
        # else:
        #    print("edge {}, {} already present".format(i, j))

    def __generate_contacts(self):
        contact_file = os.path.join(self.folder, self.pdb_name + "_contacts" + ".tsv")
        pdb_file = os.path.join(self.folder, self.pdb_name + ".pdb")
        if not os.path.exists(contact_file):
            main("--structure {} --output {} --itypes all".format(pdb_file, contact_file).split())
            assert os.path.isfile(contact_file)
            print(f"Computed Contacts for: {self.pdb_name}; saved in {contact_file}")

        # Read Contacts File
        # pdb = parsePDB(os.path.join(self.folder, self.pdb_name + ".pdb"))
        # residues_names = pdb.getResnames()
        # residues_nums = pdb.getResnums()
        # names = pdb.getNames()
        with open(contact_file, "r") as f:
            next(f)
            next(f)
            for line in f:
                linfo = line.strip().split("\t")
                interaction_type = linfo[1]
                res1 = linfo[2].split(':')[1:]
                res2 = linfo[3].split(':')[1:]

                # Select interacting Atoms
                idx_1 = None
                idx_2 = None
                for i, myatom in enumerate(self.atoms_info):
                    myatom: Atom
                    if (myatom.res_idx == int(res1[1])) and (myatom.residue == res1[0]) and (
                            str(myatom.name) == res1[2]):
                        idx_1 = int(i)
                    if (myatom.res_idx == int(res2[1])) and (myatom.residue == res2[0]) and (
                            str(myatom.name) == res2[2]):
                        idx_2 = int(i)
                    if (idx_1 is not None) and (idx_2 is not None):
                        break

                # Add edge to set of edges
                self.__add_edge(idx_1, idx_2, bond_type=interaction_type)

    def connect_inside_residues(self, group: str = None):
        # TODO connect_inside_residues
        pass

    def complete_edges(self, group: str = None):
        idx_selected = []
        for idx_1 in range(len(self.atoms_info)):
            if group is not None:
                if group in self.atoms_info[idx_1].get_atom_groups():
                    idx_selected.append(idx_1)
            else:
                idx_selected.append(idx_1)

        for i in range(len(idx_selected)):
            for j in range(i + 1, len(idx_selected)):
                self.__add_edge(idx_selected[i], idx_selected[j], group=group)

    def complete_KNN_edges(self, k: int = 3, group: str = None):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.positions)
        distances, indices = nbrs.kneighbors(self.positions)
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):
                idx_1 = int(indices[i, 0])
                idx_2 = int(indices[i, j])
                self.__add_edge(idx_1, idx_2, group=group)
        pass

    def __get_features_node(self, node: Atom):
        feat = []
        feat.extend([int(node.is_bb)])  # 0
        feat.extend([int(node.is_calpha)])  # 1
        feat.extend([int(node.is_aromatic)])  # 2
        feat.extend([int(node.is_in_ring)])  # 3
        feat.extend([node.res_idx])  # 4
        feat.extend(onek_encoding_unk(node.element, self.atoms_encoded))  # 5:5+6
        feat.extend(onek_encoding_unk(node.residue, self.residues_encoded))  # 5+6:5+6+19
        feat.extend([node.charge])
        # feat.extend(onek_encoding_unk(node.atom_type, self.atoms_ff))
        return feat

    def __get_features_edge(self,
                            edge: Edge,
                            invert: bool = False,
                            compute_angles: bool = False,
                            compute_inversion_angles: bool = False):
        feat = []
        feat.extend([edge.length])
        feat.extend([int(edge.is_covalent_bond)])
        feat.extend([int(edge.connect_calpha)])

        if invert:
            begin_idx = 'j'
            end_idx = 'i'
        else:
            begin_idx = 'i'
            end_idx = 'j'

        if compute_angles:
            if invert:
                feat.extend(edge.angles_j)
                feat.extend([0] * (self.max_angles - len(edge.angles_j)))
                feat.extend(edge.angles_i)
                feat.extend([0] * (self.max_angles - len(edge.angles_i)))
            else:
                feat.extend(edge.angles_i)
                feat.extend([0] * (self.max_angles - len(edge.angles_i)))
                feat.extend(edge.angles_j)
                feat.extend([0] * (self.max_angles - len(edge.angles_j)))
        if compute_inversion_angles:
            if invert:
                feat.extend(edge.dihedral_j)
                feat.extend([0] * (self.max_dihedrals - len(edge.dihedral_j)))
                feat.extend(edge.dihedral_i)
                feat.extend([0] * (self.max_dihedrals - len(edge.dihedral_i)))
            else:
                feat.extend(edge.dihedral_i)
                feat.extend([0] * (self.max_dihedrals - len(edge.dihedral_i)))
                feat.extend(edge.dihedral_j)
                feat.extend([0] * (self.max_dihedrals - len(edge.dihedral_j)))

        feat.extend(onek_encoding_unk(edge.type, self.edge_encoded))
        feat.extend(onek_encoding_unk(edge.features["{}_symbol".format(begin_idx)], self.atoms_encoded))
        feat.extend(onek_encoding_unk(edge.features["{}_symbol".format(end_idx)], self.atoms_encoded))
        feat.extend([edge.features["{}_charge".format(begin_idx)]])
        feat.extend([edge.features["{}_charge".format(end_idx)]])
        feat.extend([edge.features["{}_radii".format(begin_idx)]])
        feat.extend([edge.features["{}_radii".format(end_idx)]])
        feat.extend([int(edge.is_aromatic)])
        feat.extend([int(edge.is_in_ring)])
        return feat

    def get_atoms_type_encoded(self):
        res = []
        for node in self.atoms_info:
            res.append(self.atoms_ff.index(node.atom_type))
        return res

    def generate_graph(self,
                       compute_angles: bool = False,
                       compute_inversion_angles: bool = False,
                       match_features: bool = True,
                       group: str = None):
        edge_index = []
        edge_attr = []

        for edge in self.edges_info:
            edge: Edge
            edge_index.append([edge.begin_atom, edge.end_atom])
            edge_index.append([edge.end_atom, edge.begin_atom])
            feat = self.__get_features_edge(edge, False, compute_angles, compute_inversion_angles)
            edge_attr.append(feat)
            feat = self.__get_features_edge(edge, True, compute_angles, compute_inversion_angles)
            edge_attr.append(feat)

        node_x = []
        for node in self.atoms_info:
            if group is not None:
                if group in node.get_atom_groups():
                    node_x.append(self.__get_features_node(node))
            else:
                node_x.append(self.__get_features_node(node))

        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_x = torch.tensor(node_x, dtype=torch.float)
        if match_features:
            if node_x.shape[1] < edge_attr.shape[1]:
                node_x = torch.cat(
                    (node_x, torch.zeros((node_x.shape[0], edge_attr.shape[1] - node_x.shape[1]))), 1)
            elif node_x.shape[1] > edge_attr.shape[1]:
                edge_attr = torch.cat(
                    (edge_attr, torch.zeros((edge_attr.shape[0], node_x.shape[1] - edge_attr.shape[1]))), 1)
        data = Data(x=node_x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
        return data

    def download_pdb(self, pdb_code: str) -> None:
        """
        Download PDB structure from PDB

        :param pdb_code: 4 character PDB accession code
        :return: # todo impl return
        """
        # Initialise class and download pdb file
        fetchPDB(pdb_code, folder=self.folder, compressed=False)

    def set_frame(self, i):
        self.positions = self.trajectory.xyz[i]

    def get_time(self, i):
        return self.trajectory.time[i]

    def get_n_frames(self):
        return self.trajectory.n_frames

    def load_trajectory(self, dcd_file):
        sele = '(not water) and (mass > 1.5)'
        try:
            self.trajectory = md.load(dcd_file if '.dcd' in dcd_file else dcd_file + '.dcd', top=self.pdb_file)
            self.pdb_noh, idx = self.remove_from_pdb(None, sele)
            self.initialize()
            self.trajectory = self.trajectory.atom_slice(idx)
        except:
            self.trajectory = md.load(dcd_file if '.dcd' in dcd_file else dcd_file + '.dcd', top=self.pdb_noh)

    def run(self, simulation_length: Unit, n_simulations: int, unfold: bool = True):
        solute_dielectric = 1.0
        solvent_dielectric = 78.5

        # ONLY PROTEIN
        prmtop = AmberPrmtopFile(self.protein.prmtop_file)
        inpcrd = AmberInpcrdFile(self.protein.prmcrd_file, loadBoxVectors=True)
        system = prmtop.createSystem(nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1 * nanometer,
                                     constraints=HBonds,
                                     implicitSolvent=OBC2, soluteDielectric=solute_dielectric,
                                     solventDielectric=solvent_dielectric,
                                     implicitSolventSaltConc=0.15 * molar)

        system.addForce(AndersenThermostat(298.15, 1.0))
        integrator = VerletIntegrator(0.002 * picosecond)
        #integrator = LangevinIntegrator(298.15, 1.0, 0.002 * picosecond) #temperature, frictionCoeff, stepSize
        my_simulation = Simulation(prmtop.topology, system, integrator)
        if inpcrd.boxVectors is not None:
            my_simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

        trj_files = []
        if not os.path.exists(os.path.join(self.folder, "trajectories")):
            os.mkdir(os.path.join(self.folder, "trajectories"))

        for i in range(n_simulations):
            dcd_file = os.path.join(self.folder, "trajectories", self.pdb_name + '_' + str(i) + '.dcd')
            if os.path.exists(dcd_file):
                try:
                    length = md.load_dcd(dcd_file, top=self.protein.pdb_file).n_frames
                except:
                    length = md.load_dcd(dcd_file, top=self.pdb_noh).n_frames
                if length >= int(simulation_length / (0.002 * picosecond) / 5000):
                    trj_files.append(dcd_file)
                    continue
                else:
                    os.system("rm {}".format(dcd_file))
            my_simulation.reporters.clear()
            my_simulation.context.setPositions(inpcrd.positions)
            my_simulation.minimizeEnergy(maxIterations=10000, tolerance=0.5 * kilocalorie_per_mole)
            my_simulation.context.setVelocitiesToTemperature(298.15)
            if unfold:
                my_simulation.context.setParameter(AndersenThermostat.Temperature(), 600 * kelvin)
                print("Temperature is : {} K".format(
                    my_simulation.context.getParameter(AndersenThermostat.Temperature())))
                my_simulation.step(int(5 * nanosecond / (0.002 * picosecond)))
                my_simulation.context.setParameter(AndersenThermostat.Temperature(), 300 * kelvin)
                print("Temperature is : {} K".format(
                    my_simulation.context.getParameter(AndersenThermostat.Temperature())))

            print("save positions each : {}".format(0.002 * picosecond * 5000))
            reporter_dcd = DCDReporter(dcd_file, 5000)
            rep = StateDataReporter(file=sys.stdout, totalSteps=int(simulation_length / (0.002 * picosecond)),
                                    reportInterval=50000, remainingTime=True, speed=True)
            my_simulation.reporters.append(reporter_dcd)
            my_simulation.reporters.append(rep)
            print("starting simulation {}".format(i))
            my_simulation.step(int(simulation_length / (0.002 * picosecond)))
            print("finishing simulation {}".format(i))
            trj_files.append(dcd_file)

        return trj_files, int(simulation_length / (0.002 * picosecond) / 5000)

    def remove_from_pdb(self, pdb_file=None, selection=None, renumber=False):
        if pdb_file is None:
            pdb_file = self.pdb_file[:-4] + "_noh.pdb"
            os.system('cp {} {}'.format(self.pdb_file, pdb_file))
        if selection is None:
            selection = '(not water) and (mass > 1.5)'
        trj = md.load_pdb(pdb_file)
        idx = trj.trj.select(selection)
        trj = trj.atom_slice(idx)
        trj.save_pdb(pdb_file)
        if renumber:
            FileUtility.renumber_atoms(pdb_file)
        return pdb_file, idx

    def tica_analysis(self, files: str = None, images_names: str = None):
        distances_feat = pyemma.coordinates.featurizer(self.pdb_file)
        distances_feat.add_distances(
            distances_feat.pairs(distances_feat.select_Backbone(), excluded_neighbors=2), periodic=False)
        distances_data = pyemma.coordinates.load(files, features=distances_feat)
        tica = pyemma.coordinates.tica(distances_data, lag=5)
        tica_output = tica.get_output()
        tica_concatenated = np.concatenate(tica_output)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        pyemma.plots.plot_feature_histograms(
            tica_concatenated,
            ax=axes[0],
            feature_labels=['IC1', 'IC2', 'IC3', 'IC4'],
            ylog=True)
        pyemma.plots.plot_density(*tica_concatenated[:, :2].T, ax=axes[1], logscale=True)
        axes[1].set_xlabel('IC 1')
        axes[1].set_ylabel('IC 2')
        fig.tight_layout()

        if images_names is not None:
            plt.savefig(images_names + "_TICA_map.png")

        fig, axes = plt.subplots(4, 1, figsize=(12, 5), sharex=True)
        x = 0.1 * np.arange(tica_output[0].shape[0])
        for i, (ax, tic) in enumerate(zip(axes.flat, tica_output[0].T)):
            ax.plot(x, tic)
            ax.set_ylabel('IC {}'.format(i + 1))
        axes[-1].set_xlabel('time / ns')
        fig.tight_layout()
        if images_names is not None:
            plt.savefig(images_names + "_TICA_trj.png")

        cluster = pyemma.coordinates.cluster_kmeans(
            tica_output, k=75, max_iter=50, stride=10, fixed_seed=1)
        dtrajs_concatenated = np.concatenate(cluster.dtrajs)

        fig, ax = plt.subplots(figsize=(4, 4))
        pyemma.plots.plot_density(
            *tica_concatenated[:, :2].T, ax=ax, cbar=False, alpha=0.3)
        ax.scatter(*cluster.clustercenters[:, :2].T, s=5, c='C1')
        ax.set_xlabel('IC 1')
        ax.set_ylabel('IC 2')
        if images_names is not None:
            plt.savefig(images_names + "_TICA_clusters.png")
        fig.tight_layout()
