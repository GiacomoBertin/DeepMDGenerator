import mdtraj as md
import numpy as np
import pyemma
import torch
from pdbfixer import PDBFixer
from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *

from utils.ProteinGraph import ProteinGraph
from utils.dataset import DeepMDDataset


class PCA:
    def __init__(self, Data=None):
        self.data = Data
        self.eigenpair = None
        self.U = None

    def __repr__(self):
        return f'PCA({self.data})'

    def decomposition(self, X, k, center=True):
        # Center the Data using the static method within the class
        # X is a data matrix with m samples and n features
        # matmul(X, V[:, :k]) projects data to the first k principal components
        # U is m x q matrix
        # S is q - vector
        # V is n x q matrix
        U, S, V = torch.pca_lowrank(X, center=center)
        y = torch.matmul(X, V[:, :k])

        # Save variables to the class object, the eigenpair and the centered data
        self.U = U
        self.eigenpair = (V[:, :k], S)
        self.data = X
        return y

    def transform(self, X, center=False):
        if center:
            return torch.matmul(X - torch.mean(X, dim=0, keepdim=True), self.eigenpair[0])
        else:
            return torch.matmul(X, self.eigenpair[0])

    def explained_variance(self):
        # Total sum of eigenvalues (total variance explained)
        tot = sum(self.eigenpair[1])
        # Variance explained by each principal component
        var_exp = [(i / tot) for i in sorted(self.eigenpair[1], reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        return cum_var_exp


class ModelAnalysis:
    def __init__(self, database: DeepMDDataset = None):
        self.database = database

    @staticmethod
    def plot_acorr(data, xlabel='', ylabel=''):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4.9), dpi=100)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.acorr(data, data=None, maxlags=99)

    @staticmethod
    def plot_graph(xy_graphs, xlabel='', ylabel='', labels=None, file=None, legend_pos='best'):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4.9), dpi=100)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        for i, data in enumerate(xy_graphs):
            x, y = data
            if labels is not None:
                plt.plot(x, y, label=labels[i])
            else:
                plt.plot(x, y)
        if labels is not None:
            plt.legend(loc=legend_pos)
        if file is not None:
            plt.savefig(file, format='png', transparent=False)
        plt.close()

    @staticmethod
    def plot_errorbars(xyerr_graphs, xlabel='', ylabel='', labels=None, file=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(5, 4.9), dpi=100)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        clrs = sns.color_palette("hls", len(xyerr_graphs))
        for i, data in enumerate(xyerr_graphs):
            x, y, error = data
            plt.plot(x, y, label=labels[i] if labels is not None else None, color=clrs[i])
            plt.fill_between(x, y - error, y + error, alpha=0.5, color=clrs[i])

        if labels is not None:
            plt.legend()
        if file is not None:
            plt.savefig(file, format='png', transparent=False)
        plt.close()

    @staticmethod
    def plot_histos(histograms, xlabel='', ylabel='', labels=None, file=None, colors=None, nbins=20, min_x=None,
                    max_x=None, density=True):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4.9), dpi=100)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        if isinstance(histograms[0], torch.Tensor):
            if min_x is None:
                min_x = torch.cat(histograms).min().item()
            if max_x is None:
                max_x = torch.cat(histograms).max().item()

        for i, data in enumerate(histograms):
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            if len(histograms) > 1:
                plt.hist(data, bins=np.arange(min_x, max_x,
                                              (max_x - min_x) / nbins),
                         alpha=0.7, color=colors[i] if colors is not None else None,
                         label=labels[i] if labels is not None else None, density=density)
            else:
                plt.hist(data, bins=np.arange(np.min(histograms), np.max(histograms),
                                              (np.max(histograms) - np.min(histograms)) / nbins),
                         alpha=0.7, color=colors[i] if colors is not None else None,
                         label=labels[i] if labels is not None else None, density=density)
        if labels is not None:
            plt.legend()
        if file is not None:
            plt.savefig(file, format='png', transparent=False)
        plt.close()

    @staticmethod
    def plot_data(data: np.array, xy_train=None, xy_deep=None, file=None):
        x, y = data
        # Set up default x and y limits
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, MaxNLocator
        plt.figure(figsize=(5.3, 5), dpi=100)
        xlims = [min(x), max(x)]
        ylims = [min(y), max(y)]

        # Set up your x and y labels
        xlabel = '$\mathrm{PCA\\ 1}$'
        ylabel = '$\mathrm{PCA\\ 2}$'

        # Define the locations for the axes
        left, width = 0.12, 0.55
        bottom, height = 0.12, 0.55
        bottom_h = left_h = left + width + 0.02

        # Set up the geometry of the three plots
        rect_temperature = [left, bottom, width, height]  # dimensions of temp plot
        rect_histx = [left, bottom_h, width, 0.25]  # dimensions of x-histogram
        rect_histy = [left_h, bottom, 0.25, height]  # dimensions of y-histogram

        # Make the three plots
        axTemperature = plt.axes(rect_temperature)  # temperature plot
        axHistx = plt.axes(rect_histx)  # x histogram
        axHisty = plt.axes(rect_histy)  # y histogram

        # Remove the inner axes numbers of the histograms
        nullfmt = NullFormatter()
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        # Find the min/max of the data
        xmin = min(xlims)
        xmax = max(xlims)
        ymin = min(ylims)
        ymax = max(y)

        # pyemma.plots.plot_density
        # Make the 'main' temperature plot
        # Define the number of bins
        nxbins = 50
        nybins = 50
        nbins = 100

        xbins = np.linspace(start=xmin, stop=xmax, num=nxbins)
        ybins = np.linspace(start=ymin, stop=ymax, num=nybins)
        aspectratio = (1.0 * (xmax - 0) / (1.0 * ymax - 0))
        if aspectratio < 0.0:
            aspectratio = 1.0 / abs(aspectratio)
        H, xedges, yedges = np.histogram2d(y, x, bins=(ybins, xbins))

        import matplotlib.colors as pltcl

        # Plot the axes labels
        axTemperature.set_xlabel(xlabel, fontsize=10)
        axTemperature.set_ylabel(ylabel, fontsize=10)

        # Make the tickmarks pretty
        ticklabels = axTemperature.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(5)
            label.set_family('serif')

        ticklabels = axTemperature.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(5)
            label.set_family('serif')

        # Set up the plot limits
        axTemperature.set_xlim(xlims)
        axTemperature.set_ylim(ylims)

        # Set up the histogram bins
        xbins = np.arange(xmin, xmax, (xmax - xmin) / nbins)
        ybins = np.arange(ymin, ymax, (ymax - ymin) / nbins)

        # Plot the histograms

        axHistx.hist(x, bins=xbins, density=True, alpha=0.9, color='orange')
        axHisty.hist(y, bins=ybins, density=True, alpha=0.9, orientation='horizontal', color='orange')

        if xy_train is not None:
            axHistx.hist(xy_train[0], bins=xbins, alpha=0.4, density=True, color='black')
            axHisty.hist(xy_train[1], bins=ybins, alpha=0.4, density=True, orientation='horizontal', color='black')
            axTemperature.plot(xy_train[0], xy_train[1], 's', color='black', markersize=0.08, label='Train')

        if xy_deep is not None:
            axHistx.hist(xy_deep[0], bins=xbins, alpha=0.4, density=True, color='black')
            axHisty.hist(xy_deep[1], bins=ybins, alpha=0.4, density=True, orientation='horizontal', color='black')
            axTemperature.plot(xy_deep[0], xy_deep[1], 's', color='black', markersize=0.08, label='Generated')

        # Plot the temperature data
        # H += 0.01
        cax = (axTemperature.imshow(H, extent=[xmin, xmax, ymin, ymax], cmap='YlOrBr',
                                    interpolation='bicubic', origin='lower', aspect='auto', norm=pltcl.LogNorm()))
        axTemperature.legend(bbox_to_anchor=(1.5, 1.5), loc='upper right', borderaxespad=0., markerscale=10)

        # Set up the histogram limits
        axHistx.set_xlim(min(x), max(x))
        axHisty.set_ylim(min(y), max(y))

        # Make the tickmarks pretty
        ticklabels = axHistx.get_yticklabels()
        for label in ticklabels:
            label.set_fontsize(5)
            label.set_family('serif')

        # Make the tickmarks pretty
        ticklabels = axHisty.get_xticklabels()
        for label in ticklabels:
            label.set_fontsize(5)
            label.set_family('serif')

        # Cool trick that changes the number of tickmarks for the histogram axes
        axHisty.xaxis.set_major_locator(MaxNLocator(4))
        axHistx.yaxis.set_major_locator(MaxNLocator(4))

        # Save to a File
        if file is not None:
            plt.savefig(file, format='png', transparent=False)
        plt.close()

    @staticmethod
    def perform_tica_analysis(pdb, files, features: str = 'bb_torsions', lag=5):
        assert features in ['bb_torsions', 'ca_distances', 'bb_distances', 'ha_distances']
        data = None
        if features == 'bb_torsions':
            torsions_feat = pyemma.coordinates.featurizer(pdb)
            torsions_feat.add_backbone_torsions(cossin=True, periodic=False)
            data = pyemma.coordinates.load(files, features=torsions_feat)

        if features == 'ca_distances':
            distances_feat = pyemma.coordinates.featurizer(pdb)
            distances_feat.add_distances(
                distances_feat.pairs(distances_feat.select_Ca(), excluded_neighbors=2), periodic=False)
            data = pyemma.coordinates.load(files, features=distances_feat)

        if features == 'bb_distances':
            distances_feat = pyemma.coordinates.featurizer(pdb)
            distances_feat.add_distances(
                distances_feat.pairs(distances_feat.select_Backbone(), excluded_neighbors=2), periodic=False)
            data = pyemma.coordinates.load(files, features=distances_feat)

        if features == 'ha_distances':
            distances_feat = pyemma.coordinates.featurizer(pdb)
            distances_feat.add_distances(
                distances_feat.pairs(distances_feat.select_Heavy(), excluded_neighbors=2), periodic=False)
            data = pyemma.coordinates.load(files, features=distances_feat)

        tica = pyemma.coordinates.tica(data, lag=lag)
        tica_output = tica.get_output()
        tica_concatenated = np.concatenate(tica_output)
        return tica_concatenated, tica, data

    @staticmethod
    def model_run_tica(model, n_runs: int, noise_dim: int, dataset: DeepMDDataset, pdb_name: str, out_prefix,
                       max_frame=-1):
        with torch.no_grad():
            files = dataset.trj_files[pdb_name]
            pdb = os.path.join(dataset.root, pdb_name, f'{pdb_name}_noh.pdb')
            all_trj = []

            for i in range(len(files)):
                pg, trj = dataset.get_trajectory(pdb_name=pdb_name, file_id=i, frame_i=0, frame_j=-1, lag=1)
                all_trj.append(trj.detach().cpu().numpy())

            trj_hat, _, _, _, _ = model.run(torch.randn((n_runs, noise_dim), device=model.device),
                                            torch.tensor(all_trj[0][0], device=model.device), pg,
                                            save=False, recon=True)
            pg.save_dcd(trj_hat.cpu().detach().numpy(), './temp.dcd')
            res, tica, feat = ModelAnalysis.perform_tica_analysis(pdb, files, 'bb_distances', 5)
            res_hat, tica_hat, feat_hat = ModelAnalysis.perform_tica_analysis(pdb, ['./temp.dcd'], 'bb_distances', 5)
            xy = tica._transform_array(np.concatenate(np.array(feat)[:, :max_frame, :]))
            xy_hat = tica._transform_array(feat_hat)

            ModelAnalysis.plot_2d_histo(res[:, :2], xy[:, :2], f'{out_prefix}_real.png')
            ModelAnalysis.plot_2d_histo(res[:, :2], xy_hat[:, :2], f'{out_prefix}_deep.png')
            ModelAnalysis.plot_trj_as_gif(trj_hat, f'{out_prefix}_deep_d.gif', lag=int(
                trj_hat.shape[0] / 100))
            ModelAnalysis.plot_trj_as_gif(torch.tensor(all_trj[0]), f'{out_prefix}_real_d.gif', lag=int(
                all_trj[0].shape[0] / 100))

    @staticmethod
    def get_pca(pdb_name, dataset):
        files = dataset.trj_files[pdb_name]
        pdb = os.path.join(dataset.root, pdb_name, f'{pdb_name}_noh.pdb')
        all_trj_al = []
        from utils.ProteinGraph import find_rigid_alignment
        pg, trj = dataset.get_trajectory(pdb_name=pdb_name, file_id=0, frame_i=0, frame_j=-1, lag=1)
        all_trj = trj
        for i in range(1, len(files)):
            pg, trj = dataset.get_trajectory(pdb_name=pdb_name, file_id=i, frame_i=0, frame_j=-1, lag=1)
            all_trj = torch.cat([all_trj, trj])

        for i in range(all_trj.shape[0]):
            rot, tr = find_rigid_alignment(all_trj[i], all_trj[0])
            all_trj_al.append((rot.mm(all_trj[i].T)).T + tr)
        all_trj_al = torch.stack(all_trj_al, dim=0)
        pca = PCA()
        y = pca.decomposition(all_trj_al.view(all_trj_al.shape[0], -1), k=2, center=True)
        return pca, all_trj_al, y

    @staticmethod
    def compute_RMSF(trj, ref, mask=None):
        if mask is None:
            return torch.pow(trj - ref, 2.0).sum(-1).mean(dim=0)
        else:
            return torch.pow(trj[:, mask] - ref[mask], 2.0).sum(-1).mean(dim=0)

    @staticmethod
    def d_Hausdorff(trj_P, trj_Q):
        # n_frames, n_atoms, 3
        d2_matrix = torch.cdist(trj_P.view(trj_P.shape[0], -1), trj_Q.view(trj_Q.shape[0], -1))
        d_PQ = torch.max(torch.min(d2_matrix, -1)[0])
        d_QP = torch.max(torch.min(d2_matrix, -2)[0])
        return torch.max(d_PQ, d_QP)

    @staticmethod
    def model_run_pca_fin(model, n_runs: int, noise_dim: int, dataset: DeepMDDataset, pdb_name: str, out_prefix, lag=1):
        with torch.no_grad():
            files = dataset.trj_files[pdb_name]
            pdb = os.path.join(dataset.root, pdb_name, f'{pdb_name}_noh.pdb')
            prot_graph = None

            # Compute the real PCA
            real_trj = []
            conf_zer = None
            for i in range(0, len(files)):
                pg, trj = dataset.get_trajectory(pdb_name=pdb_name, file_id=i, frame_i=0, frame_j=-1, lag=lag)
                trj = trj[2:]
                if prot_graph is None:
                    prot_graph = pg
                if conf_zer is None:
                    conf_zer = trj[0]
                trj = ModelAnalysis.align_trj(trj, conf_zer, pg.get_backbone_mask())
                real_trj.append(trj)

            real_trj_stack = torch.cat(real_trj, dim=0)
            pg.save_dcd(real_trj_stack.cpu().detach().numpy(), './all.dcd')
            pg.save_dcd(conf_zer.unsqueeze(0).cpu().detach().numpy(), './ref.dcd')

            # Collect all the frames of train set in 1 tensor
            train_trj = []
            starting_conf = []

            for i in range(0, len(files)):
                pg, trj = dataset.get_trajectory(pdb_name=pdb_name, file_id=i, frame_i=0, frame_j=dataset.max_frame,
                                                 lag=lag)
                trj = trj[2:]
                starting_conf.append(trj[0])
                trj = ModelAnalysis.align_trj(trj, conf_zer, pg.get_backbone_mask())
                train_trj.append(trj)

            # trj: n_frames, n_atoms, 3
            # all_trj: n_frames * n_files, n_atoms, 3

            train_trj_stack = torch.cat(train_trj, dim=0)
            pg.save_dcd(train_trj_stack.cpu().detach().numpy(), './train.dcd')

            # Generate Fake trj
            # u_fake = mda.Universe(pdb, './deep.dcd')
            # fake_trj_stack = torch.tensor(u_fake.trajectory.get_array(), device=model.device)
            fake_trj = []
            for i in range(len(starting_conf)):
                print(i)
                trj, _, _, = model.run(torch.randn((n_runs, noise_dim), device=model.device), starting_conf[i], pg,
                                       save=False, recon=True)

                trj = ModelAnalysis.align_trj(trj, conf_zer, pg.get_backbone_mask())
                fake_trj.append(trj)

            fake_trj_stack = torch.cat(fake_trj, dim=0)
            pg.save_dcd(fake_trj_stack.cpu().detach().numpy(), './deep.dcd')

            ############################################################################################################
            ############################################  Compute distances  ###########################################
            ############################################################################################################
            distances_train_test = []
            for i, tr_i in enumerate(train_trj):
                for j, tr_j in enumerate(real_trj):
                    if j > i:
                        distances_train_test.append(ModelAnalysis.d_Hausdorff(tr_i, tr_j).item())

            distances_train_train = []
            for i, tr_i in enumerate(train_trj):
                for j, tr_j in enumerate(train_trj):
                    if j > i:
                        distances_train_train.append(ModelAnalysis.d_Hausdorff(tr_i, tr_j).item())

            distances_test_test = []
            for i, tr_i in enumerate(real_trj):
                for j, tr_j in enumerate(real_trj):
                    if j > i:
                        distances_test_test.append(ModelAnalysis.d_Hausdorff(tr_i, tr_j).item())

            ModelAnalysis.plot_histos([torch.tensor(distances_train_test),
                                       torch.tensor(distances_train_train),
                                       torch.tensor(distances_test_test)],
                                      xlabel='$\mathrm{Hausdorff\\ distance}$',
                                      labels=['$\mathrm{D(Train,\\ All)}$', '$\mathrm{D(Train,\\ Train)}$',
                                              '$\mathrm{D(All,\\ All)}$'],
                                      file=f'{out_prefix}_all_dist.png',
                                      colors=['darkorange', 'dimgray', 'red'], nbins=50)

            distances_deep_test = []
            for i, tr_i in enumerate(fake_trj):
                for j, tr_j in enumerate(real_trj):
                    if j > i:
                        distances_deep_test.append(ModelAnalysis.d_Hausdorff(tr_i, tr_j).item())

            distances_deep_train = []
            for i, tr_i in enumerate(fake_trj):
                for j, tr_j in enumerate(train_trj):
                    if j > i:
                        distances_deep_train.append(ModelAnalysis.d_Hausdorff(tr_i, tr_j).item())

            ModelAnalysis.plot_histos([torch.tensor(distances_deep_test), torch.tensor(distances_deep_train)],
                                      xlabel='$\mathrm{Hausdorff\\ distance}$',
                                      labels=['$\mathrm{D(Generated,\\ All)}$', '$\mathrm{D(Generated,\\ Train)}$'],
                                      file=f'{out_prefix}_deep_dist.png',
                                      colors=['darkorange', 'dimgray'], nbins=50)

            print(np.mean(distances_train_test), np.std(distances_train_test))
            print(np.mean(distances_train_train), np.std(distances_train_train))
            print(np.mean(distances_deep_test), np.std(distances_deep_test))
            print(np.mean(distances_deep_train), np.std(distances_deep_train))

            ############################################################################################################
            ############################################  Compute the RMSF  ############################################
            ############################################################################################################
            x = torch.arange(0, conf_zer.shape[0]).cpu().numpy()
            real_rmsf = ModelAnalysis.compute_RMSF(real_trj_stack, conf_zer).cpu().numpy()
            train_rmsf = ModelAnalysis.compute_RMSF(train_trj_stack, conf_zer).cpu().numpy()
            fake_rmsf = ModelAnalysis.compute_RMSF(fake_trj_stack, conf_zer).cpu().numpy()

            ModelAnalysis.plot_graph([[x, real_rmsf], [x, train_rmsf], [x, fake_rmsf]],
                                     ylabel='$\mathrm{RMSF}$',
                                     xlabel='$\mathrm{indexes\\ Heavy\\ Atoms}$',
                                     labels=['All Frames', 'Train', 'Generated'],
                                     file=f'{out_prefix}_deep_rmsf.png')

            ############################################################################################################
            #############################################  Compute the PCA  ############################################
            ############################################################################################################
            pca = PCA()
            real_pca = pca.decomposition(real_trj_stack[:, prot_graph.get_backbone_mask(), :].view(
                real_trj_stack.shape[0], -1), k=2, center=False)
            print(f'Explained Variance: {pca.explained_variance()}')
            d_proj = pca.transform(train_trj_stack[:, prot_graph.get_backbone_mask(), :].view(
                train_trj_stack.shape[0], -1), center=False).T
            d_hat_proj = pca.transform(fake_trj_stack[:, prot_graph.get_backbone_mask(), :].view(
                fake_trj_stack.shape[0], -1), center=False).T
            ModelAnalysis.plot_data(real_pca.T.cpu().numpy(), None, d_hat_proj.cpu().numpy(),
                                    f'{out_prefix}_pca_deep.png')
            ModelAnalysis.plot_data(real_pca.T.cpu().numpy(), d_proj.cpu().numpy(), None,
                                    f'{out_prefix}_pca_train.png')

            pca = PCA()
            real_pca = pca.decomposition(real_trj_stack.view(real_trj_stack.shape[0], -1), k=2, center=False)
            d_proj = pca.transform(train_trj_stack.view(train_trj_stack.shape[0], -1), center=False).T
            d_hat_proj = pca.transform(fake_trj_stack.view(fake_trj_stack.shape[0], -1), center=False).T
            ModelAnalysis.plot_data(real_pca.T.cpu().numpy(), None, d_hat_proj.cpu().numpy(),
                                    f'{out_prefix}_pca_deep_ha.png')
            ModelAnalysis.plot_data(real_pca.T.cpu().numpy(), d_proj.cpu().numpy(), None,
                                    f'{out_prefix}_pca_train_ha.png')

            pca = PCA()
            av_conf = torch.mean(real_trj_stack, dim=0)
            real_trj_stack = ModelAnalysis.align_trj(real_trj_stack, av_conf)
            train_trj_stack = ModelAnalysis.align_trj(train_trj_stack, av_conf)
            fake_trj_stack = ModelAnalysis.align_trj(fake_trj_stack, av_conf)
            real_pca = pca.decomposition(real_trj_stack.view(real_trj_stack.shape[0], -1), k=2, center=False)
            d_proj = pca.transform(train_trj_stack.view(train_trj_stack.shape[0], -1), center=False).T
            d_hat_proj = pca.transform(fake_trj_stack.view(fake_trj_stack.shape[0], -1), center=False).T
            ModelAnalysis.plot_data(real_pca.T.cpu().numpy(), None, d_hat_proj.cpu().numpy(),
                                    f'{out_prefix}_pca_deep_average.png')
            ModelAnalysis.plot_data(real_pca.T.cpu().numpy(), d_proj.cpu().numpy(), None,
                                    f'{out_prefix}_pca_train_average.png')

    @staticmethod
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

    @staticmethod
    def align_trj(trj_A, B, mask=None):
        for k in range(trj_A.shape[0]):
            if mask is not None:
                rot, tr = ModelAnalysis.find_rigid_alignment(trj_A[k, mask, :], B[mask, :])
            else:
                rot, tr = ModelAnalysis.find_rigid_alignment(trj_A[k], B)
            trj_A[k] = (rot.mm(trj_A[k].T)).T + tr
            del rot, tr
        return trj_A

    @staticmethod
    def compute_energy_distribution(pg: ProteinGraph, dataset: DeepMDDataset, file_dcd, minimize_maxIter=-1, file=None,
                                    labels=None, lags=None):
        prmtop_file = os.path.join(dataset.root, pg.name, f'{pg.name}.prmtop')
        prmcrd_file = os.path.join(dataset.root, pg.name, f'{pg.name}.prmcrd')
        solute_dielectric = 2.0
        solvent_dielectric = 78.5
        picosecond = pico * second
        nanometer = nano * meter

        # ONLY PROTEIN
        prmtop = AmberPrmtopFile(prmtop_file)
        inpcrd = AmberInpcrdFile(prmcrd_file, loadBoxVectors=True)
        system = prmtop.createSystem(nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1 * nanometer,
                                     constraints=HBonds,
                                     implicitSolvent=OBC2, soluteDielectric=solute_dielectric,
                                     solventDielectric=solvent_dielectric,
                                     implicitSolventSaltConc=0.15 * molar)
        plat = Platform.getPlatformByName('CUDA')
        integrator = VerletIntegrator(0.002 * picosecond)
        my_simulation = Simulation(prmtop.topology, system, integrator, platform=plat)
        print(my_simulation.context.getPlatform().getName())
        if inpcrd.boxVectors is not None:
            my_simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

        tolerance = 5 * kilojoule_per_mole
        all_energies = []
        all_sasa = []
        max_frames = 50000
        if lags is None:
            lags = [1] * len(file_dcd)
        for k, dcd in enumerate(file_dcd):
            trj: md.Trajectory = md.load_dcd(dcd, pg.pdb_noh)
            energies = []
            positions_min = None

            for i in range(0, min(trj.n_frames, max_frames), lags[k]):
                try:
                    print(i)
                    pos = trj.openmm_positions(i)
                    fixer = PDBFixer(filename=pg.pdb_noh)
                    fixer.positions = pos
                    fixer.addMissingHydrogens(7.0)
                    if not os.path.exists('temp.pdb'):
                        PDBFile.writeFile(fixer.topology, fixer.positions, open('temp.pdb', 'w'))
                    my_simulation.context.setPositions(fixer.positions)
                    if minimize_maxIter > 0:
                        my_simulation.minimizeEnergy(tolerance=tolerance, maxIterations=minimize_maxIter)
                    state: State = my_simulation.context.getState(getEnergy=True, getPositions=True)
                    energies.append(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
                    if positions_min is None:
                        positions_min = state.getPositions(asNumpy=True)
                        positions_min = positions_min.reshape((1, positions_min.shape[0], positions_min.shape[1]))
                    else:
                        pos = state.getPositions(asNumpy=True)
                        positions_min = np.append(positions_min, pos.reshape((1, pos.shape[0], pos.shape[1])), axis=0)
                    del fixer
                except Exception as e:
                    pass
                finally:
                    pass

            trj: md.Trajectory = md.load_pdb('temp.pdb')
            trj.xyz = positions_min
            sasa = md.shrake_rupley(trj)
            total_sasa = sasa.sum(axis=1)
            all_sasa.append(torch.tensor(total_sasa))
            energies = torch.tensor(energies)
            all_energies.append(energies[energies < 0])
        if file is not None:
            ModelAnalysis.plot_histos(all_energies, xlabel='$\mathrm{Energy\\ (kJ/mol)}$', labels=labels,
                                      file=file + '_energy.png',
                                      nbins=50, colors=['darkorange', 'dimgray'])
            ModelAnalysis.plot_histos(all_sasa, xlabel='$\mathrm{Total\\ SASA \\ (nm^2)}$', labels=labels,
                                      file=file + '_sasa.png',
                                      nbins=50, colors=['darkorange', 'dimgray'])

    @staticmethod
    def move_comparison(pg: ProteinGraph, dataset: DeepMDDataset, model, pdb_name, n_frames, probs,
                        out_png=None, lag=1):
        from openmmtools import cache
        from openmmtools.states import ThermodynamicState, SamplerState
        import openmmtools.integrators as integrators
        from geomloss import SamplesLoss
        from simtk import unit
        from MCMCHybrid import MCMCHybridMove, WeightedMove, PivotMove, MCMCMySampler, GHMCMove
        from scipy import signal

        ed = SamplesLoss("energy")

        def get_3n(t, t_hat, mask=None):
            t_a = ModelAnalysis.align_trj(t, torch.mean(t, dim=0), mask=mask)
            t_hat = ModelAnalysis.align_trj(t_hat, torch.mean(t, dim=0), mask=mask)
            return ed(t_a.view(t_a.shape[0], -1), t_hat.view(t_hat.shape[0], -1))

        # Get real trj
        files = dataset.trj_files[pdb_name]
        pdb = os.path.join(dataset.root, pdb_name, f'{pdb_name}.pdb')
        real_trj = []
        for i in range(0, len(files)):
            _, trj = dataset.get_trajectory(pdb_name=pdb_name, file_id=i, frame_i=0, frame_j=-1, lag=lag)
            trj = trj[2:]
            real_trj.append(trj)
        real_trj = torch.cat(real_trj, dim=0).float()
        av_conf = torch.mean(real_trj, dim=0)
        real_trj = ModelAnalysis.align_trj(real_trj, av_conf)
        mask_ha = torch.tensor(md.load_pdb(pg.pdb_noh[:-8] + '.pdb').top.select('mass > 1.5'),
                               dtype=torch.long, device=real_trj.device)

        prmtop_file = os.path.join(dataset.root, pg.name, f'{pg.name}.prmtop')
        prmcrd_file = os.path.join(dataset.root, pg.name, f'{pg.name}.prmcrd')
        solute_dielectric = 1.0
        solvent_dielectric = 78.5

        prmtop = AmberPrmtopFile(prmtop_file)
        inpcrd = AmberInpcrdFile(prmcrd_file, loadBoxVectors=True)

        system = prmtop.createSystem(nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1 * unit.nanometer,
                                     constraints=HBonds,
                                     implicitSolvent=OBC2, soluteDielectric=solute_dielectric,
                                     solventDielectric=solvent_dielectric,
                                     implicitSolventSaltConc=0.15 * molar)
        reference_platform = Platform.getPlatformByName('CUDA')
        cache.global_context_cache.platform = reference_platform
        integrator = integrators.LangevinIntegrator(temperature=298.15 * unit.kelvin,
                                                    collision_rate=1.0 / unit.picosecond,
                                                    timestep=2.0 * unit.femtosecond)
        context = Context(system, integrator, reference_platform)
        print(context.getPlatform().getName())
        # Minimize system
        context.setPositions(inpcrd.positions)
        LocalEnergyMinimizer.minimize(context)
        min_positions = context.getState(getPositions=True).getPositions()

        # Define Thermodynamic State
        thermodynamic_state = ThermodynamicState(system=system, temperature=298.15 * unit.kelvin)

        ghmc_move = GHMCMove(timestep=2.0 * unit.femtosecond, n_steps=5000, collision_rate=1.0 / unit.picosecond)
        pivot_move = PivotMove(pdb, n_steps=1)
        # disp_move = MCDisplacementMove(displacement_sigma=1. * unit.nanometer)
        # IntegratorMove(integrator=integrator, n_steps=5000)
        # HMCMove(timestep=2.0 * unit.femtoseconds, n_steps=500)
        # GHMCMove(timestep=2.0 * unit.femtosecond, n_steps=5000, collision_rate=1.0 / unit.picoseconds)
        mcmch = MCMCHybridMove(pg, model, context, n_steps=1)
        labels = []
        labels.extend(['pivot'])
        labels.extend([f'{p}' for p in probs])
        moves = []
        moves.extend([pivot_move])
        moves.extend([WeightedMove([(ghmc_move, p), (mcmch, 1.0 - p)]) for p in probs])

        all_trajectories = []
        all_energies = []
        all_sampling_efficiency = []
        n_runs = 1
        n_points = None
        torch.backends.cudnn.enabled = False
        lag_k = None
        mse = torch.nn.MSELoss()
        xy = []
        with torch.no_grad():
            for i, move in enumerate(moves):
                move_trajectories = []
                move_sampling_efficiency = []
                move_energies = []
                statistic = None
                correlations = []
                for l in range(n_runs):
                    torch.cuda.empty_cache()
                    context.setPositions(min_positions)
                    context.setVelocitiesToTemperature(298.15)
                    sampler_state = SamplerState.from_context(context)
                    sampler = MCMCMySampler(thermodynamic_state=thermodynamic_state,
                                            sampler_state=sampler_state, move=move, context=context)
                    trajectories = sampler.run(n_frames[i])
                    statistic = sampler.move.statistics
                    energies = sampler.energies
                    move_energies.append(sampler.energies)
                    del sampler, sampler_state
                    sampling_efficiency = []
                    trajectories = torch.tensor(trajectories, dtype=torch.float, device=real_trj.device)[:, mask_ha]
                    trajectories = ModelAnalysis.align_trj(trajectories, av_conf)
                    move_trajectories.append(trajectories)
                    lag_k = int((trajectories.shape[0]) / 40)
                    for k in range(2, trajectories.shape[0] + 1, lag_k):
                        torch.cuda.empty_cache()
                        ed = SamplesLoss("energy")
                        trj = trajectories[0:k]
                        e = ed(real_trj.view(real_trj.shape[0], -1), trj.view(trj.shape[0], -1))
                        sampling_efficiency.append(e.item())
                    move_sampling_efficiency.append(sampling_efficiency)
                    n_points = trajectories.shape[0] + 1
                    correlations.append(ModelAnalysis.get_autocorrelation(torch.tensor(energies), 100).cpu().numpy())

                move_sampling_efficiency = torch.tensor(move_sampling_efficiency)
                move_sampling_efficiency_mean = torch.mean(move_sampling_efficiency, dim=0)
                move_sampling_efficiency_std = torch.std(move_sampling_efficiency, dim=0)

                torch.save(torch.tensor(move_energies), f'test/{labels[i]}_energies.pt')
                move_energies_mean = np.mean(move_energies, axis=0)

                xy.append([np.arange(0, correlations[0].shape[0]), np.mean(correlations, axis=0)])
                ModelAnalysis.plot_graph(xy, xlabel='Lag', ylabel='Correlation', labels=labels, file=f'test/corr.png',
                                         legend_pos='upper right')

                move_trajectories = torch.cat(move_trajectories, dim=0)
                torch.save(move_trajectories.cpu(), f'test/{labels[i]}_trj.pt')
                torch.save(move_sampling_efficiency.cpu(), f'test/{labels[i]}_move_eff.pt')
                all_trajectories.append(move_trajectories)

                x = []
                for k in range(1, move_sampling_efficiency_mean.shape[0] + 1):
                    x.append(k * lag_k)  # / move_sampling_efficiency_mean.shape[0] * 100)
                all_sampling_efficiency.append([x,
                                                move_sampling_efficiency_mean.cpu().numpy(),
                                                move_sampling_efficiency_std.cpu().numpy()])

                print(f'statistic: {statistic}')
                if isinstance(statistic, dict):
                    print(f"efficiency: {statistic['n_accepted'] / (statistic['n_proposed'] + 1)}")
                else:
                    print(f"efficiency: {statistic[0]['n_accepted'] / (statistic[0]['n_proposed'] + 1)},"
                          f" {statistic[1]['n_accepted'] / (statistic[1]['n_proposed'] + 1)}")
                pca = PCA()
                real_pca = pca.decomposition(real_trj.view(real_trj.shape[0], -1), k=2, center=False)
                move_trajectories = ModelAnalysis.align_trj(move_trajectories, av_conf)
                d_proj = pca.transform(move_trajectories.view(move_trajectories.shape[0], -1), center=False).T
                ModelAnalysis.plot_data(real_pca.T.cpu().numpy(), None, d_proj.cpu().numpy(),
                                        file=out_png + f'_pca_{labels[i]}.png')
                print(f'Explained Variance: {pca.explained_variance()}')

                ModelAnalysis.plot_errorbars(all_sampling_efficiency, xlabel='MC Steps',
                                             ylabel='$\mathbb{E}[D_{E}(\mathbb{P}(\mathbf{x}_{t+\\tau}), \mathbb{P}(\mathbf{'
                                                    '\hat{x}}_{t+\\tau}))$',
                                             labels=labels, file=out_png + '_sampling_eff.png')

    @staticmethod
    def compute_pca(dataset: DeepMDDataset, pt_file, pdb_name, out_png,lag=1):
        files = dataset.trj_files[pdb_name]
        pdb = os.path.join(dataset.root, pdb_name, f'{pdb_name}.pdb')
        real_trj = []
        for i in range(0, len(files)):
            _, trj = dataset.get_trajectory(pdb_name=pdb_name, file_id=i, frame_i=0, frame_j=-1, lag=lag)
            trj = trj[2:]
            real_trj.append(trj)
        real_trj = torch.cat(real_trj, dim=0).float()
        av_conf = torch.mean(real_trj, dim=0)
        real_trj = ModelAnalysis.align_trj(real_trj, av_conf)
        trj = torch.load(pt_file)
        pca = PCA()
        real_pca = pca.decomposition(real_trj.view(real_trj.shape[0], -1), k=2, center=False)
        move_trajectories = ModelAnalysis.align_trj(trj, av_conf)
        d_proj = pca.transform(move_trajectories.view(move_trajectories.shape[0], -1), center=False).T
        ModelAnalysis.plot_data(real_pca.T.cpu().numpy(), None, d_proj.cpu().numpy(),
                                file=out_png)
        print(f'Explained Variance: {pca.explained_variance()}')

    @staticmethod
    def get_autocorrelation(data, n):
        M = data.shape[0]
        mean = torch.mean(data)
        corr = torch.zeros(n)
        std2 = ((data - mean) ** 2).sum() / M
        for k in range(0, n):
            if k == 0:
                corr[k] = (data - mean) @ (data - mean) / (M - k) / std2
            else:
                corr[k] = (data[:-k] - mean) @ (data[k:] - mean) / (M - k) / std2
        return torch.abs(corr)

    @staticmethod
    def move_autocorrelation(labels, n):
        xy = []

        for label in labels:
            move_energies = torch.load(f'test/{label}_energies.pt')
            corrs = []
            for i in range(1):
                data = move_energies[i]
                M = data.shape[0]
                mean = torch.mean(data)
                corr = torch.zeros(n)
                std2 = ((data - mean) ** 2).sum() / M
                for k in range(0, n):
                    if k == 0:
                        corr[k] = torch.abs(data - mean) @  torch.abs(data - mean) / (M - k) / std2
                    else:
                        corr[k] = torch.abs(data[:-k] - mean) @ torch.abs(data[k:] - mean) / (M - k) / std2
                corrs.append(corr.cpu().numpy())
            corr = np.mean(corrs, axis=0)
            xy.append([np.arange(0, n), corr])
            ModelAnalysis.plot_graph(xy, xlabel='Lag', ylabel='Correlation', labels=labels, file=f'test/corr.png',
                                     legend_pos='upper right')
