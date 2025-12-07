import itertools

import numpy as np
import pyscf
import scipy


class Hubb:
    def __init__(self, config):
        self.L = config["L"]  # number of sites
        self.N = config["N"]  # number of particles
        self.t = config["t"]  # Fermi-Hubbard hopping strength
        self.U = config["U"]  # Fermi-Hubbard on-site strength
        self.seed = config["seed"]  # seed for random couplings
        self.spin_mode = config["spin_mode"]  # number of spin modes per site
        self.coupling_method = config["couplings"]  # kind of couplings to use

        if self.spin_mode not in ["0", "1/2 spinfree", "1/2"]:
            raise ValueError("spin_mode must be \"0\", \"1/2 spinfree\" or \"1/2\"")

        self.ghf = (self.spin_mode == "1/2")

        np.random.seed(self.seed)

        self.pot_hopping, self.pot_chemical, self.pot_coupling = self.init_potentials()

        self.h, self.g, self.g_spatial = self.init_hg()
        self.C_scf = self.scf()
        # self.C_spin_block_diag, self.hf_order = self.spin_block_diagonalise_coeff()
        # self.C_spin_block_diag = np.identity(self.C_scf.shape[0])

        print("C_scf = \n{}".format(self.C_scf))
        # print("C_sbd = \n{}".format(self.C_spin_block_diag))

        self.h_MO, self.g_MO = self.calc_hg_MO()
        self.h_HF, self.g_HF = self.calc_hg_HF()

    def init_potentials(self):
        pot_hopping_up = np.zeros((self.L, self.L))
        pot_hopping_down = np.zeros((self.L, self.L))
        pot_chemical_up = np.zeros(self.L)
        pot_chemical_down = np.zeros(self.L)
        pot_coupling = np.zeros((self.L, self.L))

        M_1 = np.random.rand(self.L, self.L) * 2 - 1
        M_2 = np.random.rand(self.L, self.L)

        t = self.t
        U = self.U

        if self.coupling_method == "random":
            # hopping potentials
            pot_hopping_up = -(M_1 + M_1.T) / 2
            pot_hopping_down = pot_hopping_up
            # on-site potentials
            # pot_chemical_up = -np.random.rand(self.L)
            pot_chemical_up = np.zeros(self.L)
            pot_chemical_down = pot_chemical_up
            # two-site repulsion potentials
            pot_coupling = 8 * np.diag(np.random.rand(self.L)) - 4 * np.identity(self.L)
        elif self.coupling_method == "fermi_hubbard":
            for p in range(self.L):
                pot_hopping_up[p, (p - 1) % self.L] = -t# + 0.000001 * np.random.rand()
            pot_hopping_down = pot_hopping_up
            pot_coupling = U * np.identity(self.L)# + np.diag(np.random.rand(self.L)) * 0.0001
            # pot_coupling[1, 1] *= 2  # nonuniform U
        elif self.coupling_method == "KjS1":
            J = 1
            J02 = 0.685
            pot_hopping_up[0, 1] = -J
            pot_hopping_up[1, 2] = -J
            pot_hopping_up[0, 2] = -J02
            pot_hopping_down[0, 1] = -J
            pot_hopping_down[1, 2] = -J
            pot_hopping_down[0, 2] = +J02
            pot_coupling[0, 0] = U
            pot_coupling[2, 2] = U
        elif self.coupling_method == "KjS2":
            epsilon = -t / 10
            for p in range(self.L):
                pot_hopping_up[p, (p - 1) % self.L] = -t# + 0.000001 * np.random.rand()
            pot_hopping_down = np.copy(pot_hopping_up)
            pot_hopping_up[0, 3] += epsilon
            pot_hopping_up[1, 2] += epsilon
            pot_hopping_down[0, 3] -= epsilon
            pot_hopping_down[1, 2] -= epsilon
            pot_coupling = U * np.identity(self.L)
            # pot_coupling[:-1, :-1] = 0

        return [pot_hopping_up, pot_hopping_down], [pot_chemical_up, pot_chemical_down], pot_coupling

    def init_hg(self):
        L = self.L
        h_up = np.zeros((L, L))
        h_down = np.zeros((L, L))
        g_spatial = np.zeros((L, L, L, L))

        for p in range(L):
            h_up[p, p] += self.pot_chemical[0][p]
            h_down[p, p] += self.pot_chemical[1][p]
            for q in range(L):
                h_up[p, q] += self.pot_hopping[0][p, q]
                h_up[q, p] += self.pot_hopping[0][p, q]
                h_down[p, q] += self.pot_hopping[1][p, q]
                h_down[q, p] += self.pot_hopping[1][p, q]

        h = scipy.linalg.block_diag(h_up, h_down)

        for p in range(L):
            for r in range(L):
                g_spatial[p, p, r, r] = self.pot_coupling[p, r]

        g = np.einsum("klij", g_spatial)  # swap first two axes with last two
        g = np.kron(np.identity(2), g)  # only p, q indices with equal spin are nonzero (np.kron uses last axes)
        g = np.einsum("klij", g)  # swap axes back
        g = np.kron(np.identity(2), g)  # only r, s indices with equal spin are nonzero

        return h, g, g_spatial

    def calc_hg_MO(self):
        h = np.copy(self.h)
        g = np.copy(self.g)

        C = self.C_spin_block_diag

        h_MO = C.T @ h @ C
        g_MO = np.einsum("PM, RK, QN, SL, PQRS -> MNKL", C, C, C, C, g)

        return h_MO, g_MO

    def calc_hg_HF(self):
        L = self.L
        h_HF = np.zeros((2 * L, 2 * L))
        g_HF = np.zeros((2 * L, 2 * L, 2 * L, 2 * L))

        for P in range(2 * L):
            # g_c_P = self.g_MO[P, P, :self.N, :self.N]  # coulumb part
            g_c_P = self.g_MO[P, P, self.hf_order[:self.N], :][:, self.hf_order[:self.N]]  # coulumb part
            # g_e_P = self.g_MO[P, :self.N, :self.N, P]  # exchange part
            g_e_P = self.g_MO[P, self.hf_order[:self.N], :, P][:, self.hf_order[:self.N]]  # exchange part
            eps_P = self.h_MO[P, P] + np.sum(g_c_P) - np.sum(g_e_P)
            # h_HF[P, P] = eps_P
            h_HF[P, P] = eps_P - self.U / 2  # standard definition is without this U / 2

        return h_HF, g_HF

    def scf(self):
        # IMPORTANT: identity is assumed for overlap matrix
        mol_hubb = pyscf.gto.M()
        mol_hubb.nelectron = self.N
        mol_hubb.incore_anyway = True

        if not self.ghf:  # RHF
            hf_hubb = pyscf.scf.RHF(mol_hubb)
            hf_hubb.get_hcore = lambda *args: self.h[:self.L, :self.L]
            hf_hubb.get_ovlp = lambda *args: np.identity(self.L)
        else:
            hf_hubb = pyscf.scf.GHF(mol_hubb)
            hf_hubb.get_hcore = lambda *args: self.h
            hf_hubb.get_ovlp = lambda *args: np.identity(2 * self.L)

        hf_hubb._eri = pyscf.ao2mo.restore(8, self.g_spatial, self.L)  # 8-fold symmetry
        hf_hubb.init_guess = '1e'
        hf_hubb.kernel()

        print("orbital energies (PySCF): {}".format(hf_hubb.mo_energy))

        C = hf_hubb.mo_coeff
        orb_norms = np.diag(C.T @ C)  # norms of unnormalised MOs
        C = C / np.sqrt(orb_norms)

        if not self.ghf:  # RHF
            C2 = scipy.linalg.block_diag(C, C)
            index_list = [val for pair in zip(range(self.L), range(self.L, 2 * self.L)) for val in pair]
            C = C2[:, index_list]

        return C

    def spin_block_diagonalise_coeff(self):
        # This assumes no orbital mixing between spin up and down, so that spin up MOs will be on the top L rows and
        # spin down MOs on the bottom L rows
        C_up = self.C_scf[:self.L, :]  #
        arg_up = np.argwhere(np.all(C_up[..., :] == 0, axis=0))
        C_up = np.delete(C_up, arg_up, axis=1)
        C_down = self.C_scf[self.L:, :]
        arg_down = np.argwhere(np.all(C_down[..., :] == 0, axis=0))
        C_down = np.delete(C_down, arg_down, axis=1)

        # When spin orbitals are reordered the order of ascending orbital energies is lost, so recreate that order here
        indices_down = arg_up.flatten()  # arg_up is list of columns where up has all zeroes i.e. these are down spins
        indices_up = arg_down.flatten()
        indices = np.concatenate((indices_up, indices_down))
        hf_order = np.array([]).astype(int)
        for i in range(len(indices)):
            hf_order = np.append(hf_order, np.where(indices == i)[0][0])

        return scipy.linalg.block_diag(C_up, C_down), hf_order
