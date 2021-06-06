# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# from graphein.construct_graphs import ProteinGraph
from utils.dataset import *
from torch_geometric.nn import GCNConv, ARGVA
import prody as pd
import mdshare
import mdtraj as md
from copy import deepcopy
import models as mm
from Bio import pairwise2 as pw2
from utils.dataset import Dataset
#from DeepGGRNN import RNN, Decoder, train
#from DeepMarkovModelLight import DeepMarkovModel, train

discriminator_train_epochs = 10
model_train_epochs = 100000

import multiprocessing as mp
from models_1 import *


def test(dataset: DeepMDDataset):
    # TODO maybe use only a subset and reconstruction loss
    pass


def sistema_i_cazzo_di_files():
    root = "/home/giacomo/Documents/DeepMD"
    pdb_wrong = os.path.join(root, 'backup/pentapeptide/pentapeptide-impl-solv_sel.pdb')
    files_ff_wrong = [os.path.join(root, 'backup/pentapeptide/trajectories_dcd', 'pentapeptide_{}.dcd'.format(i))
                      for i in range(0, 24)]
    files_ff_corrected = [os.path.join(root, 'backup/pentapeptide/trajectories_new', 'pentapeptide_{}.dcd'.format(i))
                          for i in range(0, 24)]

    for k, file in enumerate(files_ff_wrong):
        trj: md.Trajectory = md.load(file, top=pdb_wrong)
        for i in range(trj.n_frames):
            old = deepcopy(trj.xyz[i])
            trj.xyz[i, 11] = old[7]
            trj.xyz[i, 10] = old[8]
            trj.xyz[i, 7] = old[10]
            trj.xyz[i, 8] = old[11]
        trj.save_dcd(files_ff_corrected[k], force_overwrite=True)


def sequences_match(first_seq, second_seq):
    global_align = pw2.align.globalxx(first_seq, second_seq)
    seq_length = min(len(first_seq), len(second_seq))
    matches = global_align[0][2]
    percent_match = (matches / seq_length)
    return percent_match


"""pdb = parsePDB(actual_pdb, folder=database_folder, compressed=False)
                sequence = pdb.getSequence()
                if sequences_match() > 90.0:
                    pass"""


def sistema_i_cazzo_di_files_2():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/MicroModel"
    pdb_list = ["1agi", "1bfg", "1bj7", "1bsn", "1chn", "1cqy", "1csp", "1czt", "1emr", "1fas",
                "1fvq", "1gnd", "1i6f", "1il6", "1j5d", "1jli", "1k40", "1kte", "1kxa", "1lit",
                "1lki", "1nso", "1ooi", "1opc", "1pdo", "1pht", "1sdf", "1sp2", "1sur", "2hvm", "1ubq", "2gbl"]
    pcz_wrong = [os.path.join(root, pdb + '.pcz') for pdb in pdb_list]
    unzip = "/home/giacomo/Programs/pcasuite-linux-x86_64/pcasuite/pcaunzip"

    for k, pdb_name in enumerate(pdb_list):
        root_pdb = os.path.join(root, pdb_name)
        file_pdb_0 = os.path.join(root_pdb, pdb_name + '_temp.pdb')
        file_pdb_1 = os.path.join(root_pdb, pdb_name + '.pdb')
        file_x = os.path.join(root_pdb, pdb_name + '.x')
        if not os.path.exists(root_pdb):
            os.mkdir(root_pdb)
        if not os.path.exists(os.path.join(root_pdb, "trajectories")):
            os.mkdir(os.path.join(root_pdb, "trajectories"))
        os.system(unzip + " -i {} -o {} --pdb".format(os.path.join(root, pdb_name + '.pcz'), file_pdb_0))
        os.system(unzip + " -i {} -o {}".format(os.path.join(root, pdb_name + '.pcz'), file_x))

        pdb_lines = []
        with open(file_pdb_0, "r") as f:
            for line in f:
                if ("MODEL" in line) and ("2" in line):
                    break
                else:
                    pdb_lines.append(line)
        with open(file_pdb_0, "w") as f:
            f.writelines(pdb_lines)
        os.system("pdb4amber -i {} -o {}".format(file_pdb_0, file_pdb_1))
        os.system("rm {}_*".format(file_pdb_1[:-4]))
        trj: md.Trajectory = md.load_pdb(file_pdb_1)
        num_atoms = trj.n_atoms

        coords = []
        with open(file_x, "r") as f:
            for line in f:
                x = [float(x) for x in line.split()]
                coords.extend(x)
        coords = np.array(coords) / 10.0
        coords = coords.reshape([-1, num_atoms, 3])
        trj.xyz = coords
        trj.save_dcd(os.path.join(root_pdb, "trajectories", pdb_name + '.dcd'))


max_num_res = 200


class pdb_item:
    def __init__(self, pdb_name, my_sequence):
        self.name = pdb_name
        self.sequence = my_sequence

    def __eq__(self, other):
        if self.name == other.name:
            return True

        match = sequences_match(self.sequence, other.sequence)
        if match > 0.95:
            return True
        else:
            return False


def compute_file(file, root, pdb_valid):
    root_temp = os.path.join(root, "temp")
    pcz_folder = os.path.join(root, "pcz")
    root_pdb = os.path.join(root, "pdb")

    unzip = "/home/giacomo/Programs/pcasuite-linux-x86_64/pcasuite/pcaunzip"
    try:
        # print os.path.join(subdir, file)
        idx = file.split('.')[0]
        file_pdb_0 = os.path.join(root_temp, idx + '_temp.pdb')
        file_pdb_1 = os.path.join(root_temp, idx + '.pdb')
        file_x = os.path.join(root_temp, idx + '.x')
        os.system(unzip + " -i {} -o {} --pdb".format(os.path.join(pcz_folder, file), file_pdb_0))
        pdb_lines = []
        with open(file_pdb_0, "r") as f:
            for line in f:
                if ("MODEL" in line) and ("2" in line):
                    break
                else:
                    pdb_lines.append(line)
        with open(file_pdb_0, "w") as f:
            f.writelines(pdb_lines)
        os.system("pdb4amber -i {} -o {}".format(file_pdb_0, file_pdb_1))
        os.system("rm {}_*".format(file_pdb_1[:-4]))
        pdb = pd.parsePDB(file_pdb_1)
        n_res = pdb.numResidues()
        if n_res < max_num_res:
            sequence = pdb.getSequence()
            item = pdb_item(idx, sequence)  # pd.blastPDB(pdb).getBest()['pdb_id'], sequence)
            root_data = os.path.join(root, "database")
            for other in pdb_valid:
                if other == item:
                    root_data = os.path.join(root, "data")
                    item.name = other.name.lower()
                    break

            if not os.path.exists(os.path.join(root_data, item.name)):
                os.mkdir(os.path.join(root_data, item.name))
            os.system("cp {} {}".format(file_pdb_1, os.path.join(root_data, item.name, item.name + ".pdb")))
            trj: md.Trajectory = md.load_pdb(file_pdb_1)
            num_atoms = trj.n_atoms
            os.system(unzip + " -i {} -o {}".format(os.path.join(pcz_folder, file), file_x))
            coords = []
            with open(file_x, "r") as f:
                for line in f:
                    x = [float(x) for x in line.split()]
                    coords.extend(x)
            coords = np.array(coords) / 10.0
            coords = coords.reshape([-1, num_atoms, 3])
            trj.xyz = coords
            trj.save_dcd(os.path.join(root_data, item.name, item.name + '.dcd'))
            os.system("rm {}".format(os.path.join(root_temp, idx + "*")))
        return 0
    except:
        return 1


def sistema_i_cazzo_di_files_3():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL"
    unzip = "/home/giacomo/Programs/pcasuite-linux-x86_64/pcasuite/pcaunzip"

    pdb_list = "/home/giacomo/Documents/DeepMD/train_MoDEL/pdb_list.txt"
    pdb_valid = []

    with open(pdb_list, "r") as f:
        for line in f:
            try:
                pdb_name = line.split()[0].lower()
                pdb_file = pd.fetchPDB(pdb_name, folder=os.path.join(root, "pdb"), compressed=False)
                pdb = pd.parsePDB(pdb_file)
                sequence = pdb.getSequence()
                n_res = pdb.numResidues()
                if n_res < max_num_res:
                    pdb_valid.append(pdb_item(pdb_name, sequence))
            except:
                pass

    root_temp = os.path.join(root, "temp")
    pcz_folder = os.path.join(root, "pcz")
    root_pdb = os.path.join(root, "pdb")
    root_data = os.path.join(root, "data")
    if not os.path.exists(root_temp):
        os.mkdir(root_temp)
    print(len(pdb_valid))

    for subdir, dirs, files in os.walk(pcz_folder):
        pool = mp.Pool(processes=4)
        results = [pool.apply_async(compute_file, args=(file, root, pdb_valid)) for file in files]
        output = [p.get() for p in results]
        if 1 in output:
            print("Messed something")


def genera_i_fottuti_tensori():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL"
    root_temp = os.path.join(root, "temp")
    pcz_folder = os.path.join(root, "pcz")
    root_pdb = os.path.join(root, "pdb")
    root_data = os.path.join(root, "data")
    root_database = os.path.join(root, "database")
    folders = []
    for subdir, dirs, files in os.walk(root_data):
        name = subdir.split(os.sep)[-1]
        if os.path.exists(os.path.join(subdir, name + '.pdb')) and os.path.exists(os.path.join(subdir, name + '.dcd')):
            folders.append([name, subdir, os.path.join(subdir, name + '.dcd'), os.path.join(subdir, name + '.pdb')])

    for folder in folders:
        if (not os.path.exists(os.path.join(folder[1], 'tensors', 'edges_length_{}_1.pt'.format(folder[0])))):
            try:
                DeepMDDataset.process_pdb_dcd_file(folder[3], folder[2], folder[1], folder[0],
                                                   save_distance_matrix=False)
            except:
                pass
    # pool = mp.Pool(processes=1)
    # results = [pool.apply_async(DeepMDDataset.process_pdb_dcd_file,
    #                            args=(folder[3], folder[2], folder[1], folder[0])) for folder in folders]
    # output = [p.get() for p in results]
    pass


def main_train3():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL"
    root_database = os.path.join(root, "train_penta")
    root_test = os.path.join(root, "test_penta")

    latent_dimesion = 100
    decoder = Decoder(latent_dim=16,
                      noise_dimension=8,
                      mlp_hiddens=[32, 16],
                      hiddens=[16, 8],
                      output_dim=1)
    model = DeepMarkovModel(nodes_features=30,
                            edges_features=30,
                            hiddens_phi=[40, 64, 32],
                            latent_dimension=16,
                            decoder=decoder)

    database = DeepMDDataset(root=root_database, lag=1, save_memory_mode=True, log_file='train.log')
    test_set = DeepMDDataset(root=root_test, lag=1, save_memory_mode=True, log_file='test.log')
    # pentapeptide has 5000 frames, each of 500 ns saved each 100 ps, MoDEL 10 ns saved each 1 ps
    database.load_folder(root_database)
    test_set.load_folder(root_test)
    database.get_trj = True
    test_set.get_trj = True
    database.trj_frames = 500
    test_set.trj_frames = 500
    train(database, model, test_set)


def process():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL"
    root_database = os.path.join(root, "train_penta")
    database = DeepMDDataset(root=root_database, lag=1, save_memory_mode=True)
    database.process(root_database)


def test_visualization():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL/pentapeptide"
    database = DeepMDDataset(root=root, lag=1, save_memory_mode=True)
    # pentapeptide has 5000 frames, each of 500 ns saved each 100 ps, MoDEL 10 ns saved each 1 ps
    database.load_folder(root)
    database.get_trj = True
    database.trj_frames = 5000
    database.max_frame_train = 4500
    database.train()
    all_trj = []
    pdb = mdshare.fetch('pentapeptide-impl-solv.pdb', working_directory='data')
    files = mdshare.fetch('pentapeptide-*-500ns-impl-solv.xtc', working_directory='data')

    for i in range(len(files)-1):
        _, trj = database.get_trajectory(pdb_name='pentapeptide', file_id=i, frame_i=0, frame_j=5000)
        trj = trj.detach().cpu().numpy()
        all_trj.append(trj)
    tica, t = DataAnalysis.perform_tica_analysis(all_trj)
    _, trj = database.get_trajectory(pdb_name='pentapeptide', file_id=5, frame_i=0, frame_j=5000)
    xy = t._transform_array(trj.detach().cpu().numpy())
    DataAnalysis.plot_2d_histo(tica[:, :2], xy[:, :2], 'test_5000.png')


"""    distances_feat = pyemma.coordinates.featurizer(pdb)
    distances_feat.add_distances(
        distances_feat.pairs(distances_feat.select_Heavy(), excluded_neighbors=2), periodic=False)
    distances_data = pyemma.coordinates.load(files, features=distances_feat)
    tica_concatenated = DataAnalysis.perform_tica_analysis(distances_data)
    DataAnalysis.plot_2d_histo(tica_concatenated[:, :2], 'test_1.png')"""

def test_visualization1():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL/pentapeptide"
    database = DeepMDDataset(root=root, lag=1, save_memory_mode=True)
    # pentapeptide has 5000 frames, each of 500 ns saved each 100 ps, MoDEL 10 ns saved each 1 ps
    database.load_folder(root)
    database.get_trj = True
    database.trj_frames = 5000
    database.max_frame_train = 4500
    database.train()
    all_trj = []
    pdb_name = '1809'
    for i in range(len(database.trj_files[pdb_name])):
        _, trj = database.get_trajectory(pdb_name=pdb_name, file_id=i, frame_i=0, frame_j=5000)
        trj = trj.detach().cpu().numpy()
        all_trj.append(trj)
    tica = DataAnalysis.perform_tica_analysis(all_trj)
    DataAnalysis.plot_2d_histo(tica[:, :2], 'test_1809.png')
    return 0


def main_train4():
    os.environ['PATH'] += os.pathsep + '/home/giacomo/Programs/miniconda3/bin/'
    root = "/home/giacomo/Documents/DeepMD/train_MoDEL"
    root_database = os.path.join(root, "train_penta")
    root_test = os.path.join(root, "test_penta")

    model = DeepMarkovModel(nodes_features=30,
                            hiddens_decoder=[64, 128, 64, 32, 16, 8])

    database = DeepMDDataset(root=root_database, lag=10, save_memory_mode=True, log_file='train.log')
    test_set = DeepMDDataset(root=root_test, lag=10, save_memory_mode=True, log_file='test.log')
    # pentapeptide has 5000 frames, each of 500 ns saved each 100 ps, MoDEL 10 ns saved each 1 ps
    database.load_folder(root_database)
    test_set.load_folder(root_test)
    database.get_trj = True
    test_set.get_trj = True
    database.trj_frames = 30
    test_set.trj_frames = 30
    train(database, model, test_set)


if __name__ == '__main__':
    process()
