import os
import pyscf
import scipy

import numpy as np


root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))


class ES:
    def __init__(self, config):
        self.atoms_config_file = open("{}/input/{}.dat".format(root_dir, config["atoms_config"]))
        self.basis_set = config["basis_set"]

        self.atoms_config = self.atoms_config_file.read()

        self.spin_mode = config["spin_mode"]  # number of spin modes per site

        if self.spin_mode not in ["0", "1/2 spinfree", "1/2"]:
            raise ValueError("spin_mode must be \"0\", \"1/2 spinfree\" or \"1/2\"")

        self.ghf = False#(self.spin_mode == "1/2")

        self.mol = self.init_mol()  # molecule object
        self.N = self.mol.nelectron  # number of electrons
        self.E_nuc = pyscf.gto.Mole.energy_nuc(self.mol)

        self.h_AO, self.g_AO, self.S_AO = self.init_integrals()

        self.L = self.h_AO.shape[0]

        self.h, self.g = self.init_hg()
        self.C_scf = self.scf()
        self.C_spin_block_diag, self.hf_order = self.spin_block_diagonalise_coeff()
        # self.C_spin_block_diag = np.identity(self.C_scf.shape[0])

        # print("C_scf = \n{}".format(self.C_scf))
        # print("C_sbd = \n{}".format(self.C_spin_block_diag))

        self.h_MO, self.g_MO = self.calc_hg_MO()
        self.h_HF, self.g_HF = self.calc_hg_HF()

    def init_mol(self):
        mol = pyscf.gto.Mole()
        mol.unit = "Angstrom"
        mol.atom = self.atoms_config
        mol.basis = self.basis_set
        mol.build()

        return mol

    def init_integrals(self):
        h_AO = self.mol.intor("int1e_kin") + \
                    self.mol.intor("int1e_nuc")  # one-electron integrals in spatial AO basis
        g_AO = self.mol.intor("int2e")  # two-electron integrals in spatial AO basis
        S_AO = self.mol.intor("int1e_ovlp")  # overlap matrix

        return h_AO, g_AO, S_AO

    def init_hg(self):
        L = self.L

        h = scipy.linalg.block_diag(self.h_AO, self.h_AO)

        g = np.einsum("klij", self.g_AO)  # swap first two axes with last two
        g = np.kron(np.identity(2), g)  # only p, q indices with equal spin are nonzero (np.kron uses last axes)
        g = np.einsum("klij", g)  # swap axes back
        g = np.kron(np.identity(2), g)  # only r, s indices with equal spin are nonzero

        return h, g

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
            h_HF[P, P] = eps_P

        return h_HF, g_HF

    def scf(self):
        if not self.ghf:  # RHF
            hf_es = pyscf.scf.RHF(self.mol)
        else:
            hf_es = pyscf.scf.GHF(self.mol)

        hf_es.kernel()

        print("orbital energies (PySCF): {}".format(hf_es.mo_energy))

        C = hf_es.mo_coeff
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