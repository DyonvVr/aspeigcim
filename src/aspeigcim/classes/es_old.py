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

        self.mol = self.init_mol()  # molecule object
        self.N = self.mol.nelectron  # number of electrons
        self.E_nuc = pyscf.gto.Mole.energy_nuc(self.mol)

        self.h_AO, self.g_AO, self.S_AO = self.init_integrals()
        self.h_MO, self.g_MO, self.rhf = self.scf()

        self.k = self.h_MO.shape[0]

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

    def init_test_integrals(self):
        scf_test_dir = os.path.join(root_dir, "input", "scf_test")
        t_AO_test_list = np.loadtxt(os.path.join(scf_test_dir, "t.dat"))
        v_AO_test_list = np.loadtxt(os.path.join(scf_test_dir, "v.dat"))
        s_AO_test_list = np.loadtxt(os.path.join(scf_test_dir, "s.dat"))
        g_AO_test_list = np.loadtxt(os.path.join(scf_test_dir, "eri.dat"))

        t_AO_test = np.zeros((7, 7))
        for line in t_AO_test_list:
            i = int(line[0]) - 1
            j = int(line[1]) - 1
            t_AO_test[i, j] = t_AO_test[j, i] = line[2]
        v_AO_test = np.zeros((7, 7))
        for line in v_AO_test_list:
            i = int(line[0]) - 1
            j = int(line[1]) - 1
            v_AO_test[i, j] = v_AO_test[j, i] = line[2]
        h_AO_test = t_AO_test + v_AO_test

        s_AO_test = np.zeros((7, 7))
        for line in s_AO_test_list:
            i = int(line[0]) - 1
            j = int(line[1]) - 1
            s_AO_test[i, j] = s_AO_test[j, i] = line[2]

        g_AO_test = np.zeros((7, 7, 7, 7))
        for line in g_AO_test_list:
            i = int(line[0]) - 1
            j = int(line[1]) - 1
            k = int(line[2]) - 1
            l = int(line[3]) - 1
            g_AO_test[i, j, k, l] = g_AO_test[i, j, l, k] = g_AO_test[j, i, k, l] = g_AO_test[j, i, l, k] = \
                g_AO_test[k, l, i, j] = g_AO_test[l, k, i, j] = g_AO_test[k, l, j, i] = g_AO_test[l, k, j, i] = line[4]

        return h_AO_test, g_AO_test, s_AO_test

    def scf(self, uhf=False):
        hf = self.mol.RHF() if not uhf else self.mol.UHF()
        hf.kernel()

        # molecular orbital coefficients
        C = hf.mo_coeff

        if not uhf:
            return C, C
        return C[0], C[1]

        # # one-electron integrals in spatial MO basis: h_MO[i, j] = (i|h|j)
        # h_MO = C.T @ self.h_AO @ C
        # # two-electron integrals in spatial MO basis: g_MO[i, j, k, l] = (ij|kl)
        # g_MO = np.einsum("mp, kr, nq, ls, mnkl -> pqrs", C, C, C, C, self.g_AO)

        # h_AO_test, g_AO_test, S_AO_test = self.init_test_integrals()
        #
        # h_AO_s = np.kron(h_AO_test, np.identity(2))
        #
        # g_AO_s = np.einsum("klij", g_AO_test)  # swap first two axes with last two
        # g_AO_s = np.kron(g_AO_s, np.identity(2))  # only p, q indices with equal spin are nonzero (np.kron uses last axes)
        # g_AO_s = np.einsum("klij", g_AO_s)  # swap axes back
        # g_AO_s = np.kron(g_AO_s, np.identity(2))  # only r, s indices with equal spin are nonzero
        #
        # S_AO_s = np.kron(S_AO_test, np.identity(2))
        #
        # C_mnospin = self.scf(h_AO_s, g_AO_s, S_AO_s)
        #
        # print("C_pyscf = \n{}".format(C))
        # print("C_mnospin = \n{}".format(C_mnospin))
        #
        # return h_MO, g_MO, hf

    def get_hg_MO(self):
        return self.h_MO, self.g_MO

    def getSCFEnergy(self):
        return self.rhf.kernel()

    def getFCIEnergy(self):
        return pyscf.fci.FCI(self.rhf)

    def orbital_energy(self, p):
        g_c_p = self.g_MO[p, p, :int(self.N / 2), :int(self.N / 2)]  # coulumb part
        g_e_p = self.g_MO[p, :int(self.N / 2), :int(self.N / 2), p]  # exchange part
        return self.h_MO[p, p] + 2 * np.sum(g_c_p) - np.sum(g_e_p)  # note this only works for even N!!

    def orbital_energy_two_elec_corrected(self, p):
        g_c_p = self.g_MO[p, p, :int(self.N / 2), :int(self.N / 2)]  # coulumb part
        g_e_p = self.g_MO[p, :int(self.N / 2), :int(self.N / 2), p]  # exchange part
        return self.h_MO[p, p] + (1 / 2) * (2 * np.sum(g_c_p) - np.sum(g_e_p))  # note this only works for even N!!

    def get_hg_HF(self, two_elec_shifted=False):
        k = self.k
        h_HF = np.zeros((k, k))
        g_HF = np.zeros((k, k, k, k))

        two_elec_factor = 1 / 2 if two_elec_shifted else 1

        for p in range(k):
            g_c_p = self.g_MO[p, p, :int(self.N / 2), :int(self.N / 2)]  # coulumb part
            g_e_p = self.g_MO[p, :int(self.N / 2), :int(self.N / 2), p]  # exchange part
            eps_p = self.h_MO[p, p] + two_elec_factor * (2 * np.sum(g_c_p) - np.sum(g_e_p))  # note this only works for even N!!
            h_HF[p, p] = eps_p

        return h_HF, g_HF

    def scf_manual(self, h, g, S=None):
        L = h.shape[0]
        sm = 1
        if S is None:
            S = np.identity(L)

        print("S = \n{}".format(S))

        kappa, D = np.linalg.eigh(S)
        S_is = D @ np.diag(kappa ** -0.5) @ D.T.conj()
        # S_is = scipy.linalg.fractional_matrix_power(S, -0.5)  # inverse square root

        np.random.seed(23)
        # P = np.random.randn(L, L)  # density matrix
        # P = 2 * np.kron(np.random.randn(L // 2, L // 2), np.identity(2))
        # P = np.zeros((L, L))
        P = np.identity(L)
        P = (P + P.T.conj()) / 2
        P_prev = P + np.inf
        print("P = \n{}".format(P))

        C = np.zeros(L)  # orbital rotation matrix
        e = 0  # energy
        e_prev = np.inf

        print("P.shape = {}".format(P.shape))
        print("g.shape = {}".format(g.shape))

        while np.linalg.norm(P - P_prev) > 1e-14 and np.abs(e - e_prev) > 1e-14:
            f = h + 0.5 * (sm * np.einsum("kl, mnkl -> mn", P, g) - np.einsum("kl, mknl -> mn", P, g))

            # print("f = \n{}".format(f))
            # _, C = scipy.linalg.eigh(f, S)
            _, Cp = np.linalg.eigh(S_is @ f @ S_is)
            C = S_is @ Cp

            # print("C = \n{}".format(C))

            C_occ = C[:, :self.N // sm]
            P_prev = P
            P = 2 * np.matmul(C_occ, C_occ.T)

            # print("P = \n{}".format(P))

            e_prev = e
            e = sm * 0.25 * np.einsum("mn, mn ->", P, h + f)  # + E_nuc

            # print("e = {}".format(e))

            # break

        e = sm * 0.25 * np.einsum("mn, mn ->", P, h + f) + 8.002367061811
        print("e = {}".format(e))

        # normalisation
        orb_norms = np.diag(C.T @ S @ C)  # norms of unnormalised MOs
        C = C / np.sqrt(orb_norms)

        # print("C_hubb = \n{}".format(C))

        return C