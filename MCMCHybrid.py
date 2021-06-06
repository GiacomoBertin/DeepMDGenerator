from abc import ABC

from openmmtools import mcmc, utils
from Markov import EdGE, noise_dim
import simtk.unit as unit
from simtk.openmm.app import AmberPrmtopFile, AmberInpcrdFile, CutoffNonPeriodic, HBonds, OBC2, Simulation
from simtk.openmm import Platform, VerletIntegrator, State, LocalEnergyMinimizer, Context
import numpy as np
from utils.ProteinGraph import ProteinGraph
import torch
import os
import mdtraj as md
from pdbfixer import PDBFixer
from scipy.spatial.transform import Rotation as R
import abc
import copy
from tqdm import tqdm


class GHMCMove(mcmc.BaseIntegratorMove):

    def __init__(self, timestep=1.0 * unit.femtosecond, collision_rate=20.0 / unit.picoseconds,
                 n_steps=1000, **kwargs):
        super(GHMCMove, self).__init__(n_steps=n_steps, **kwargs)
        self.timestep = timestep
        self.collision_rate = collision_rate
        self.n_accepted = 0  # Number of accepted steps.
        self.n_proposed = 0  # Number of attempted steps.

    @property
    def fraction_accepted(self):
        """Ratio between accepted over attempted moves (read-only).

        If the number of attempted steps is 0, this is numpy.NaN.

        """
        if self.n_proposed == 0:
            return np.NaN
        # TODO drop the casting when stop Python2 support
        return float(self.n_accepted) / self.n_proposed

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted, n_proposed=self.n_proposed)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value['n_accepted']
        self.n_proposed = value['n_proposed']

    def reset_statistics(self):
        """Reset the internal statistics of number of accepted and attempted moves."""
        self.n_accepted = 0
        self.n_proposed = 0

    def apply(self, thermodynamic_state, sampler_state):
        """Apply the GHMC MCMC move.

        This modifies the given sampler_state. The temperature of the
        thermodynamic state is used.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use when applying the MCMC move.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to. This is modified.

        """
        # Explicitly implemented just to have more specific docstring.
        prec_pos = sampler_state.positions
        super(GHMCMove, self).apply(thermodynamic_state, sampler_state)
        diff2 = (sampler_state.positions - prec_pos).value_in_unit(unit.nanometer) ** 2
        mse = np.sqrt(diff2.sum() / diff2.reshape(-1).__len__())
        return mse > 0.001

    def __getstate__(self):
        serialization = super(GHMCMove, self).__getstate__()
        serialization['timestep'] = self.timestep
        serialization['collision_rate'] = self.collision_rate
        serialization.update(self.statistics)
        return serialization

    def __setstate__(self, serialization):
        super(GHMCMove, self).__setstate__(serialization)
        self.timestep = serialization['timestep']
        self.collision_rate = serialization['collision_rate']
        self.statistics = serialization

    def _get_integrator(self, thermodynamic_state):
        """Implement BaseIntegratorMove._get_integrator()."""
        # Store lastly generated integrator to collect statistics.
        return mcmc.integrators.GHMCIntegrator(temperature=thermodynamic_state.temperature,
                                               collision_rate=self.collision_rate,
                                               timestep=self.timestep)

    def _after_integration(self, context, thermodynamic_state):
        """Implement BaseIntegratorMove._after_integration()."""
        integrator = context.getIntegrator()

        # Accumulate acceptance statistics.
        ghmc_global_variables = {integrator.getGlobalVariableName(index): index
                                 for index in range(integrator.getNumGlobalVariables())}
        n_accepted = integrator.getGlobalVariable(ghmc_global_variables['naccept'])
        n_proposed = integrator.getGlobalVariable(ghmc_global_variables['ntrials'])
        self.n_accepted += n_accepted
        self.n_proposed += n_proposed


class MetropolizedMove(mcmc.MCMCMove):
    def __init__(self, atom_subset=None, context_cache=None):
        self.n_accepted = 0
        self.n_proposed = 0
        self.atom_subset = atom_subset
        self.context_cache = context_cache

    @property
    def statistics(self):
        """The acceptance statistics as a dictionary."""
        return dict(n_accepted=self.n_accepted, n_proposed=self.n_proposed)

    @statistics.setter
    def statistics(self, value):
        self.n_accepted = value['n_accepted']
        self.n_proposed = value['n_proposed']

    def apply(self, thermodynamic_state, sampler_state):
        """Apply a metropolized move to the sampler state.

        Total number of acceptances and proposed move are updated.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to apply the move.
        sampler_state : openmmtools.states.SamplerState
           The initial sampler state to apply the move to. This is modified.

        """
        timer = utils.Timer()
        benchmark_id = 'Applying {}'.format(self.__class__.__name__)
        timer.start(benchmark_id)

        # Check if we have to use the global cache.
        if self.context_cache is None:
            context_cache = mcmc.cache.global_context_cache
        else:
            context_cache = self.context_cache

        # Create context, any integrator works.
        context, unused_integrator = context_cache.get_context(thermodynamic_state)

        # Compute initial energy. We don't need to set velocities to compute the potential.
        # TODO assume sampler_state.potential_energy is the correct potential if not None?
        sampler_state.apply_to_context(context, ignore_velocities=True)
        initial_energy = thermodynamic_state.reduced_potential(context)

        # Handle default and weird cases for atom_subset.
        if self.atom_subset is None:
            atom_subset = slice(None)
        elif not isinstance(self.atom_subset, slice) and len(self.atom_subset) == 1:
            # Slice so that initial_positions (below) will have a 2D shape.
            atom_subset = slice(self.atom_subset[0], self.atom_subset[0] + 1)
        else:
            atom_subset = self.atom_subset

        # Store initial positions of the atoms that are moved.
        # We'll use this also to recover in case the move is rejected.
        if isinstance(atom_subset, slice):
            # Numpy array when sliced return a view, they are not copied.
            initial_positions = copy.deepcopy(sampler_state.positions[atom_subset])
        else:
            # This automatically creates a copy.
            initial_positions = sampler_state.positions[atom_subset]

        # Propose perturbed positions. Modifying the reference changes the sampler state.
        proposed_positions = self._propose_positions(initial_positions)

        # Compute the energy of the proposed positions.
        sampler_state.positions[atom_subset] = proposed_positions
        sampler_state.apply_to_context(context, ignore_velocities=True)
        proposed_energy = thermodynamic_state.reduced_potential(context)

        # Accept or reject with Metropolis criteria.
        if proposed_energy < -2000:
            sampler_state.positions[atom_subset] = initial_positions
            self.n_proposed += 1
            return False
        delta_energy = proposed_energy - initial_energy
        if (not np.isnan(proposed_energy) and
                (delta_energy <= 0.0 or np.random.rand() < np.exp(-delta_energy))):
            self.n_accepted += 1
            accepted = True
        else:
            # Restore original positions.
            sampler_state.positions[atom_subset] = initial_positions
            accepted = False
        self.n_proposed += 1

        # Print timing information.
        timer.stop(benchmark_id)
        # timer.report_timing()
        return accepted

    def __getstate__(self):
        if self.context_cache is None:
            context_cache_serialized = None
        else:
            context_cache_serialized = utils.serialize(self.context_cache)
        serialization = dict(atom_subset=self.atom_subset, context_cache=context_cache_serialized)
        serialization.update(self.statistics)
        return serialization

    def __setstate__(self, serialization):
        self.atom_subset = serialization['atom_subset']
        if serialization['context_cache'] is None:
            self.context_cache = None
        else:
            self.context_cache = utils.deserialize(serialization['context_cache'])
        self.statistics = serialization

    @abc.abstractmethod
    def _propose_positions(self, positions):
        """Return new proposed positions.

        These method must be implemented in subclasses.

        Parameters
        ----------
        positions : nx3 numpy.ndarray
            The original positions of the subset of atoms that these move
            applied to.

        Returns
        -------
        proposed_positions : nx3 numpy.ndarray
            The new proposed positions.

        """
        pass


class MCMCMySampler(mcmc.MCMCSampler):
    def __init__(self, context, **kwargs):
        super(MCMCMySampler, self).__init__(**kwargs)
        self.energies = []
        self.context = context

    def run(self, n_iterations=1):
        trajectory = []
        pbar = tqdm(total=n_iterations, colour='white')
        while len(trajectory) < n_iterations:
            accepted = self.move.apply(self.thermodynamic_state, self.sampler_state)
            if accepted:
                # print(len(trajectory))
                pos = self.sampler_state.positions.value_in_unit(unit.nanometer)
                self.sampler_state.apply_to_context(self.context)
                trajectory.append(pos)
                pbar.update(1)
                self.energies.append(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        pbar.close()
        return trajectory


class WeightedMove(mcmc.MCMCMove):
    def __init__(self, move_set, context_cache=None):
        self.move_set = move_set
        if context_cache is not None:
            for move, weight in self.move_set:
                move.context_cache = context_cache

    @property
    def statistics(self):
        """The statistics of all moves as a list of dictionaries."""
        stats = [None for _ in range(len(self.move_set))]
        for i, (move, weight) in enumerate(self.move_set):
            try:
                stats[i] = move.statistics
            except AttributeError:
                stats[i] = {}
        return stats

    @statistics.setter
    def statistics(self, value):
        for i, (move, weight) in enumerate(self.move_set):
            if hasattr(move, 'statistics'):
                move.statistics = value[i]

    def apply(self, thermodynamic_state, sampler_state):
        """Apply one of the MCMC moves in the set to the state.

        The probability that a move is picked is given by its weight.

        Parameters
        ----------
        thermodynamic_state : openmmtools.states.ThermodynamicState
           The thermodynamic state to use to propagate dynamics.
        sampler_state : openmmtools.states.SamplerState
           The sampler state to apply the move to.

        """
        moves, weights = zip(*self.move_set)
        for i in range(10):
            try:
                move = np.random.choice(np.arange(0, len(self.move_set)), p=weights)
                accepted = self.move_set[move][0].apply(thermodynamic_state, sampler_state)
                return accepted
            except:
                pass

    def __getstate__(self):
        serialized_moves = [utils.serialize(move) for move, _ in self.move_set]
        weights = [weight for _, weight in self.move_set]
        return dict(moves=serialized_moves, weights=weights)

    def __setstate__(self, serialization):
        serialized_moves = serialization['moves']
        weights = serialization['weights']
        self.move_set = [(utils.deserialize(move), weight)
                         for move, weight in zip(serialized_moves, weights)]

    def __str__(self):
        return str(self.move_set)

    def __iter__(self):
        return self.move_set


class PivotMove(MetropolizedMove):
    def __init__(self, pdb, n_steps: int = 1, max_rot: float = 2.5, **kwargs):
        super(PivotMove, self).__init__(**kwargs)
        self.trj: md.Trajectory = md.load_pdb(pdb)
        self.c_mask = self.trj.top.select('name C CA')
        self.n_steps = n_steps
        self.max_rot = max_rot

    def __get_random_c(self):
        return np.random.choice(self.c_mask[1:])

    def _propose_positions(self, initial_positions):
        """Implement MetropolizedMove._propose_positions for apply()"""
        positions = initial_positions
        for i in range(self.n_steps):
            c = self.__get_random_c()
            positions = self.__rotate_atoms(c, positions)
        return positions

    def __rotate_atoms(self, c, positions):
        positions_unit = positions.unit
        resid = self.trj.top.atom(c).residue.index
        x_initial = positions.value_in_unit(positions_unit)[self.trj.top.select(f'resid >= {resid}')]
        x_initial_mean = positions.value_in_unit(positions_unit)[c]

        # Generate a random rotation matrix.
        rotation_matrix = self.generate_random_rotation_matrix()

        x_proposed_sub = (rotation_matrix * np.matrix(x_initial - x_initial_mean).T).T + x_initial_mean
        x_proposed = positions.value_in_unit(positions_unit)
        x_proposed[self.trj.top.select(f'resid >= {resid}')] = x_proposed_sub
        return unit.Quantity(x_proposed, positions_unit)

    def generate_random_rotation_matrix(self):
        """Return a random 3x3 rotation matrix.

        Returns
        -------
        Rq : 3x3 numpy.ndarray
            The random rotation matrix.

        """
        return R.from_euler('zxy', (np.random.rand(3) - 0.5) * self.max_rot, degrees=True).as_matrix()
        # q = PivotMove._generate_uniform_quaternion()
        # return PivotMove._rotation_matrix_from_quaternion(q)


class MCMCHybridMove(MetropolizedMove):
    def __init__(self, protein_graph: ProteinGraph, model: EdGE, context: Context, n_steps: int, **kwargs):
        super(MCMCHybridMove, self).__init__(**kwargs)
        self.protein_graph = protein_graph
        self.model = model
        self.device = model.device
        self.context = context
        self.tolerance = 10 * unit.kilojoule_per_mole
        self.max_iter = 0
        self.fixer = PDBFixer(filename=self.protein_graph.pdb_noh)
        self.top = self.fixer.topology
        self.mask_ha = torch.tensor(md.load_pdb(self.protein_graph.pdb_noh[:-8] + '.pdb').top.select('mass > 1.5'),
                                    dtype=torch.long, device=model.device)
        self.n_steps = n_steps

    def displace_positions(self, positions):
        starting_position = torch.tensor(positions.value_in_unit(unit.nanometer),
                                         device=self.device).float()[self.mask_ha]
        for i in range(self.n_steps):
            noise = torch.randn((noise_dim,), device=self.device)
            starting_position = self.model(noise, starting_position, self.protein_graph, recon=True)
        starting_position = starting_position.detach().cpu().numpy() * unit.nanometer
        return starting_position

    def __getstate__(self):
        serialization = super(MCMCHybridMove, self).__getstate__()
        serialization['protein_graph'] = self.protein_graph
        serialization['model'] = self.model
        serialization['context'] = self.context
        serialization['tolerance'] = self.tolerance
        serialization['max_iter'] = self.max_iter
        serialization['fixer'] = self.fixer
        serialization['n_steps'] = self.n_steps
        return serialization

    def __setstate__(self, serialization):
        super(MCMCHybridMove, self).__setstate__(serialization)
        self.protein_graph = serialization['protein_graph']
        self.model = serialization['model']
        self.context = serialization['context']
        self.tolerance = serialization['tolerance']
        self.max_iter = serialization['max_iter']
        self.fixer = serialization['fixer']
        self.n_steps = serialization['n_steps']

    def _propose_positions(self, initial_positions):
        """Implement MetropolizedMove._propose_positions for apply()."""
        for i in range(5):
            try:
                torch.cuda.empty_cache()
                new_positions_ha = self.displace_positions(initial_positions)
                self.fixer.topology = self.top
                self.fixer.positions = new_positions_ha
                self.fixer.addMissingHydrogens(7.0)
                self.context.setPositions(self.fixer.positions)
                LocalEnergyMinimizer.minimize(context=self.context, tolerance=self.tolerance,
                                              maxIterations=self.max_iter)
                new_pos = self.context.getState(getPositions=True).getPositions(asNumpy=True)
                self.context.setPositions(initial_positions)
                # self.context.setVelocitiesToTemperature(298.15)
                return new_pos
            except BaseException as e:
                pass
        raise RuntimeError('Deep Move Failed')
