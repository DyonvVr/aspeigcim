import itertools
import math
import os

import numpy as np
import openfermion
import scipy.linalg
import scipy.special
import scipy.sparse.linalg
import time

# from .chamil import CHamil

# np.random.seed(0)

root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))


class ASPEigCIM:
    def __init__(self, config):
        self.name = config["name"]
        self.N = config["N"]
        self.spin_mode = config["spin_mode"]  # no spin, spin-1/2 spinfree or spin-1/2
        self.h_final = config["h_final"]
        self.g_final = config["g_final"]
        self.h_initial = config["h_initial"]
        self.g_initial = config["g_initial"]
        self.C_sbd = config["C_sbd"]  # spin block diagonal MO coefficient matrix
        self.evo_mode = config["evolution_mode"]  # whether we use the nqf or qf, or evolve directly
        self.ds = config["ds"]
        self.T = config["T"]

        print("N = {}".format(self.N))

        if self.spin_mode not in ["0", "1/2 spinfree", "1/2"]:
            raise ValueError("spin_mode must be \"0\", \"1/2 spinfree\" or \"1/2\"")

        if self.spin_mode == "0":
            self.spin_levels = 1
        else:
            self.spin_levels = 2

        self.LL = self.h_final.shape[0]  # number of spin orbitals
        self.L = int(self.LL / 2)  # number of spatial orbitals

        self.d = int(scipy.special.binom(self.LL, self.N))  # dimension of 2-electron Fock subspace for K spin orbitals

        self.S_z = None
        self.S_plus = None
        self.S_minus = None
        self.S_sq = None
        if self.spin_levels == 2:
            self.S_z = self.build_S_z()  # (self.d x self.d) csc matrix
            self.S_plus = self.build_S_plus()  # (self.d x self.d) csc matrix
            self.S_minus = self.build_S_minus()  # (self.d x self.d) csc matrix
            self.S_sq = self.build_S_squared()  # (self.d x self.d) csc matrix

        self.G_final   = self.build_interaction_tensor(self.h_final, self.g_final)
        self.G_initial = self.build_interaction_tensor(self.h_initial, self.g_initial)

        # self.chamil = CHamil(self.k, self.N, self.spin_modes)

        self.H_initial = self.build_hamiltonian(self.build_F(self.G_initial), print_symbolic_op=False)  # (self.d x self.d) csc matrix
        # print("H_initial = \n{}".format(self.H_initial.toarray()))
        # print("S_sq = \n{}".format(self.S_sq.toarray()))
        self.initial_eigenstates = self.diagonalise_with_spin_configuration(self.H_initial)[1]
        self.initial_ground_state = self.initial_eigenstates[:, 0]

        self.is_H_initial_zero = 0#np.isclose(scipy.sparse.linalg.norm(self.H_initial), 0)

        self.H_final = self.build_hamiltonian(self.build_F(self.G_final), print_symbolic_op=False)  # (self.d x self.d) csc matrix
        self.final_eigenstates = self.diagonalise_with_spin_configuration(self.H_final)[1]
        self.final_ground_state = self.final_eigenstates[:, 0]

        print("H_final = \n{}".format(np.real(self.H_final.toarray())))
        print("H_initial = \n{}".format(np.real(self.H_initial.toarray())))
        print("H_residual = \n{}".format(np.real((self.H_final - self.H_initial).toarray())))

        # print("H_initial_eigs = {}".format(np.linalg.eigh(np.real(self.H_initial.toarray()))[0]))
        # print("H_final_eigs = {}".format(np.linalg.eigh(np.real(self.H_final.toarray()))[0]))
        # print("initial ground state = \n{}".format(self.initial_ground_state))
        # print("final ground state = \n{}".format(self.final_ground_state))

        psi0_init = self.initial_ground_state
        # print("<S_z>(initial ground state) = {}".format(psi0_init.T.conj() @ self.S_z.toarray() @ psi0_init))
        # print("<S_+>(initial ground state) = {}".format(psi0_init.T.conj() @ self.S_plus.toarray() @ psi0_init))
        # print("<S_->(initial ground state) = {}".format(psi0_init.T.conj() @ self.S_minus.toarray() @ psi0_init))
        # print("<S^2>(initial ground state) = {}".format(psi0_init.T.conj() @ self.S_sq.toarray() @ psi0_init))

        # psi1_init = self.initial_eigenstates[:, 1]
        # print("initial first state = \n{}".format(psi1_init))
        # print("<S^z>(initial first state) = {}".format(psi1_init.T.conj() @ self.S_z.toarray() @ psi1_init))
        # print("<S^+>(initial first state) = {}".format(psi1_init.T.conj() @ self.S_plus.toarray() @ psi1_init))
        # print("<S^->(initial first state) = {}".format(psi1_init.T.conj() @ self.S_minus.toarray() @ psi1_init))
        # print("<S^2>(initial first state) = {}".format(psi1_init.T.conj() @ self.S_sq.toarray() @ psi1_init))

        psi0_fin = self.final_ground_state
        # print("final ground state =\n{}".format(psi0_fin))
        # print("<S_z>(final ground state) = {}".format(psi0_fin.T.conj() @ self.S_z.toarray() @ psi0_fin))
        # print("<S^2>(final ground state) = {}".format(psi0_fin.T.conj() @ self.S_sq.toarray() @ psi0_fin))

        # print("final ground state energy = {}".format(self.final_ground_state @ self.H_final.toarray() @ self.final_ground_state))

        self.H_res_terms, self.lambda_res = self.build_H_res_terms()
        self.num_H_res_terms = len(self.H_res_terms)

        # print("H_res_terms =")
        # for term in self.H_res_terms:
        #     print(np.real(term.toarray()))

        self.evolution_result = None

    def run(self):
        # return
        self.evolution_result = self.adiabatic_evolution(T=self.T)

    def build_interaction_tensor(self, h, g):
        L = self.L

        # four-index version of one-electron integral tensor
        if self.spin_mode in ["0", "1/2 spinfree"]:
            # if spin 0, the spin down component of h is completely ignored
            # if spin 1/2 spinfree, h_up == h_down and a spin-compressed G is used
            w = (np.einsum("pq, rs -> pqrs", h[:self.L, :self.L], np.identity(L)) +
                 np.einsum("pq, rs -> pqrs", np.identity(L), h[:self.L, :self.L])) / (self.N - 1)
            G = (w + g[:self.L, :self.L, :self.L, :self.L]) / 2  # combined one- and two-electron integral tensor
            # G = (g[:self.L, :self.L, :self.L, :self.L]) / 2  # combined one- and two-electron integral tensor
        else:  # if self.spin_mode == "1/2":
            w = (np.einsum("PQ, RS -> PQRS", h, np.identity(2 * L)) +
                 np.einsum("PQ, RS -> PQRS", np.identity(2 * L), h)) / (self.N - 1)
            G = (w + g) / 2  # combined one- and two-electron integral tensor

        return G

    def diagonalise_with_spin_configuration(self, H_csc):
        H = H_csc.toarray()

        # return np.linalg.eigh(H)

        if self.spin_levels == 1:
            return np.linalg.eigh(H)

        # if particles are spin-carrying and [H, S^2] == 0, return eigenstates which are also pure spin states
        S_sq = self.S_sq.toarray()

        if not np.isclose(np.linalg.norm(H @ S_sq - S_sq @ H), 0):
            return np.linalg.eigh(H)
            # raise ValueError("H and S^2 do not commute")

        # _, psi = np.linalg.eigh((S_sq + np.identity(self.d)) @ H)
        # 1000 * np.identity is to make sure zero energies don't cause impure spin states. Careful with large energies!!
        H_common = (H + 1000 * np.identity(self.d)) @ (S_sq + np.identity(self.d))
        _, psi = np.linalg.eigh(H_common)

        e = np.diag(psi.T.conj() @ H @ psi)
        argsort = np.argsort(e)

        return e[argsort], psi[:, argsort]

    def evolution_coeff(self, k, M, s, n=0):
    # k is term index, M is total no of terms, s is scaled evolution time, n is degree
        z = self.is_H_initial_zero  # 0 or 1
        x = (M - z) * s - (k - z)

        if (x <= 0):
            return 0
        if (x >= 1):
            return 1

        return (1 - np.cos(np.pi * x)) ** 2 / 4

        # if (n == 0):
        #     return x
        # if (n == 1):
        #     return 3 * x ** 2 - 2 * x ** 3

    def build_H_res_terms(self):
        # if direct interpolation, just return entire residual hamiltonian and a fake eigenvalue 1
        if self.evo_mode == "direct":
            F_res = self.build_F(self.G_final - self.G_initial)  # residual F matrix
            return [self.build_hamiltonian(F_res)], [1]
        else:
            return self.build_H_res_terms_from_eig()

    def build_H_res_terms_from_eig(self):  # pre-build hamiltonian terms
        # build N-body terms (in order of increasing eigenvalue)
        F_res_terms = []
        H_res_terms = []
        evo_config_str = "{}_{}".format(self.name, self.evo_mode)

        hterms_dir_path = "{}/hterms/{}".format(root_dir, evo_config_str)

        F_res = self.build_F(self.G_final - self.G_initial)  # residual F matrix

        lambda_res, U_res = np.linalg.eigh(F_res)  # F = U @ diag(l) @ U†, eigvals sorted in ascending order

        # print("F_res = \n{}".format(F_res))
        print("lambda_res = {}".format(lambda_res))
        # print("U_res = \n{}".format(U_res))
        print("len(lambda_res) = {}".format(len(lambda_res)))

        LLd = len(lambda_res)  # = LL(LL - 1)/2

        # if os.path.isdir(hterms_dir_path):
        if False:
            print("built hamiltonian terms found")
            for k in range(LLd):
                info = "loading hamiltonian terms... ({} of {})".format(k + 1, LLd)
                print("\r" + info + " " * 3, end="")
                H_k = scipy.sparse.load_npz(hterms_dir_path + "/h{}.npz".format(k))
                H_res_terms.append(H_k)
            lambda_res_ordered = np.loadtxt(hterms_dir_path + "/l.txt")
            print("done")
        else:
            # os.mkdir(hterms_dir_path)
            print("no hamiltonian terms found for {}".format(evo_config_str))

            for k in range(LLd):
                info = "building hamiltonian terms... ({} of {})".format(k + 1, LLd)
                print("\r" + info + " " * 3, end="")
                F_k = self.build_F_k(lambda_res, U_res, k)
                F_res_terms.append(F_k)
                H_k = self.build_hamiltonian(F_k)
                H_res_terms.append(H_k)
                # scipy.sparse.save_npz(hterms_dir_path + "/h{}.npz".format(i), H_i)
            # np.savetxt(hterms_dir_path + "/l.txt", lambda_res)
            print("done")

        print("len(H_res_terms) = {}".format(len(H_res_terms)))

        lambda_res_ordered = np.copy(lambda_res)

        index_groups = []

        # standard ordering
        for k in range(LLd):
            index_groups.append([k])

        # small to large ordering
        argsort = np.arange(LLd)
        # large to small ordering
        # argsort = np.arange(LLd)[::-1]
        # random ordering
        # argsort = np.random.permutation(np.arange(LLd))
        print("argsort = {}".format(argsort))

        index_groups_new = []
        for k in argsort:
            index_groups_new.append(index_groups[k])
        index_groups = index_groups_new
        lambda_res_ordered = lambda_res_ordered[argsort]

        # index_groups = [list(range(48))] + [[i] for i in range(48, 64)]
        # index_groups = [[i] for i in np.where(np.logical_not(np.isclose(lambda_res_ordered, 0)))[0]]
        # print("index_groups_filtered = {}".format(index_groups))
        # index_groups = [[24]] + [[25, 26, 27]]
        # index_groups = [[0]] + [[1, 2, 3]]

        # actually put terms in right order
        H_res_terms_combined_and_ordered = []
        for index_group in index_groups:
            # H_res_term = scipy.sparse.csc_matrix((self.d, self.d))
            H_res_term = scipy.sparse.csc_matrix((self.d, self.d))
            for index in index_group:
                H_res_term += H_res_terms[index]
            H_res_terms_combined_and_ordered.append(H_res_term)

        # and return
        return H_res_terms_combined_and_ordered, lambda_res_ordered

    def compute_H_increment(self, s, s_):
        H_increment = scipy.sparse.csc_matrix((self.d, self.d))
        M = self.num_H_res_terms

        # H(s_) - H(s) = sum_k (f_k(s_) - f_k(s)) H_k
        for k in range(M):
            coeff_diff = self.evolution_coeff(k, M, s_) - self.evolution_coeff(k, M, s)
            if coeff_diff != 0:
                H_increment += coeff_diff * self.H_res_terms[k]

        return H_increment

    # T = inf means explicit diagonalisation
    def adiabatic_evolution(self, T=np.inf):
        print("starting walk")

        S_sq = self.S_sq
        S_z = self.S_z

        M = len(self.H_res_terms)  # number of hamiltonian terms
        ds = self.ds
        I_s = np.linspace(0, 1, int(1 / ds) + 1)

        op_norms = np.zeros(M)
        for i in range(M):
            op_norms[i] = np.linalg.norm(self.H_res_terms[i].toarray(), ord=2)
        # print("op_norms/lambdas = \n{}".format(op_norms / np.abs(self.l)))

        T_red = T * M / np.sum(np.power(self.lambda_res, 2))  # reduced time
        # T_red = T * M / np.sum(np.power(op_norms, 2))  # reduced time
        # T_red = T * M / np.sum(1 - np.isclose(self.l, 0, atol=1e-3))  # reduced time

        # print("l = \n{}".format(self.l))
        print("np.sum(np.square(self.l)) = {}".format(np.sum(np.square(self.lambda_res))))
        print("T_red = {}".format(T_red))

        # plotted stuff
        energies = np.zeros((len(I_s), self.d))
        final_ground_state_overlaps = np.zeros((len(I_s), self.d))
        S_sq_evs = np.zeros((len(I_s), self.d))
        S_z_evs = np.zeros((len(I_s), self.d))
        initial_ground_state_overlaps = np.zeros((len(I_s), self.d))
        t = np.zeros(len(I_s))

        # single value outputs
        sum_abs_dH_sq = 0

        psi = None
        H = None

        time_start = time.time()
        for s_index in range(len(I_s)):
            s = s_index * ds

            if s == 0:  # here, psi is still the initial eigenstate
                if self.is_H_initial_zero:  # inintial hamiltonian is zero hamiltonian
                    H = self.H_res_terms[0]
                    print("zero hamiltonian")
                else:
                    H = self.H_initial
                e, psi = self.diagonalise_with_spin_configuration(H)
                self.initial_eigenstates = psi
                self.initial_ground_state = psi[:, 0]
                t[0] = 0
            else:
                k = int(math.floor((s - ds) * M))
                # T_k = T_red * self.l[k] ** 2
                # T_k = T_red * op_norms[k] ** 2
                T_k = T * (1 - np.isclose(self.lambda_res[k], 0, atol=1e-7))
                # T_k = T
                t[s_index] = t[s_index - 1] + T_k * ds

                dH = self.compute_H_increment(s - ds, s)
                H += dH
                sum_abs_dH_sq += np.linalg.norm(dH.toarray(), ord=2) ** 2 / ds

                if T == np.inf:
                    e, psi = self.diagonalise_with_spin_configuration(H)
                else:  # numerically integrate TDSE
                    U = scipy.linalg.expm(-1j * T_k * H * ds)
                    psi = U @ psi
                    e = np.real(np.diag(psi.T.conj() @ H @ psi))  # real anyway, just to prevent ComplexWarnings

            # keep track of interesting stuff
            energies[s_index, :] = np.real(e)  # real anyway, just to prevent ComplexWarnings
            final_ground_state_overlaps[s_index, :] = np.square(np.abs(psi.T.conj() @ self.final_ground_state))
            initial_ground_state_overlaps[s_index, :] = np.square(np.abs(psi.T.conj() @ self.initial_ground_state))

            if self.spin_levels == 2:
                S_sq_evs[s_index, :] = np.real(np.diag(psi.T.conj() @ S_sq.toarray() @ psi))  # real anyway
                S_z_evs[s_index, :] = np.real(np.diag(psi.T.conj() @ S_z.toarray() @ psi))  # real anyway

            time_end = time.time()
            info = "s = {:.3f} (time elapsed: {})".format(s, time_end - time_start)
            print("\r" + info + " " * 16, end="")
            # print(info)
        print()

        print("||H_after_evo - H_final|| = {}".format(np.linalg.norm((H - self.H_final).toarray(), ord=2)))

        project_onto_spin_sector = False
        if self.spin_levels == 2 and project_onto_spin_sector:
            # project onto singlets or triplets
            s_sq = 0  # projecting onto singlets
            # s_sq = 2  # projecting onto triplets

            num_singlets = 0
            for m in range(self.d):
                if np.isclose(S_sq_evs[0, m], s_sq, atol=1e-5):
                    num_singlets += 1

            energies_singlet_projected = np.zeros((0, num_singlets))
            spins_singlet_projected = np.zeros((0, num_singlets))
            final_ground_state_overlaps_singlet_projected = np.zeros((0, num_singlets))
            initial_ground_state_overlaps_singlet_projected = np.zeros((0, num_singlets))

            for x in range(len(energies)):
                singlet_indices_x = []
                for m in range(self.d):
                    if np.isclose(S_sq_evs[x, m], s_sq, atol=1e-5):
                        singlet_indices_x.append(m)

                energies_singlet_projected = np.vstack((energies_singlet_projected, energies[x, singlet_indices_x]))
                spins_singlet_projected = np.vstack((spins_singlet_projected, S_sq_evs[x, singlet_indices_x]))
                final_ground_state_overlaps_singlet_projected = \
                    np.vstack(
                        (final_ground_state_overlaps_singlet_projected, final_ground_state_overlaps[x, singlet_indices_x]))
                initial_ground_state_overlaps_singlet_projected = \
                    np.vstack((initial_ground_state_overlaps_singlet_projected,
                               initial_ground_state_overlaps[x, singlet_indices_x]))

            return energies_singlet_projected, \
                   spins_singlet_projected, \
                   final_ground_state_overlaps_singlet_projected, \
                   initial_ground_state_overlaps_singlet_projected, \
                   t
            # end of project onto singlets or triplets

        result =\
        {
            "energies": energies,
            "final_ground_state_overlaps": final_ground_state_overlaps,
            "initial_ground_state_overlaps": initial_ground_state_overlaps,
            "S_sq_evs": S_sq_evs,
            "S_z_evs": S_z_evs,
            "sum_abs_dH_sq": sum_abs_dH_sq,
            "t": t
        }

        return result

    def project_FO_csc_onto_particle_number_sector(self, H_FO_csc):
        d = H_FO_csc.shape[0]  # == 2^(2 * self.L)
        ld = int(np.log2(d))

        # correct particle number indices
        N_particle_indices = []
        for I in range(d):
            binI = format(I, '0{}b'.format(ld))
            if binI.count("1") == self.N:
            # if binI[:-int(ld / 2)].count("1") == self.N / 2 and binI[-int(ld / 2):].count("1") == self.N / 2:
                N_particle_indices.append(I)

        # most significant bits first as we read Slater determinant from left to right
        N_particle_indices = N_particle_indices[::-1]

        H_FO_csr = H_FO_csc.tocsr()
        H_FO_csr = H_FO_csr[N_particle_indices, :]
        H_FO_csc = H_FO_csr.tocsc()
        H_FO_csc = H_FO_csc[:, N_particle_indices]

        return H_FO_csc

    def convert_FO_to_csc(self, op_FO):
        op_FO_csc = openfermion.get_sparse_operator(op_FO)
        op_FO_csc_projected = self.project_FO_csc_onto_particle_number_sector(op_FO_csc)

        if op_FO_csc_projected.shape == (0, 0):
            return scipy.sparse.csc_matrix((self.d, self.d))

        return op_FO_csc_projected

    def build_hamiltonian(self, F, print_symbolic_op=False):  # nonquadratic factorisation, OpenFermion
        # note: F could be any LL(LL - 1)/2 x LL(LL - 1)/2 matrix at this point
        # operator expressed in fermion operators; call below makes sure OF produces right dimension

        H_FO = openfermion.FermionOperator("{maxorb}^ {maxorb}^ {maxorb} {maxorb}".format(maxorb=self.LL - 1), 1e-8)

        FO_string = "{P}^ {R}^ {S} {Q}"

        # if self.spin_mode == "0":
        #     for p, q, r, s in itertools.product(range(self.L), repeat=4):
        #         H_FO += openfermion.FermionOperator(FO_string.format(P=p, Q=q, R=r, S=s), G[p, q, r, s])
        # elif self.spin_mode == "1/2 spinfree":
        #     for p, q, r, s in itertools.product(range(self.L), repeat=4):
        #         for sigma, tau in itertools.product(range(2), repeat=2):
        #             P = p + self.L * sigma
        #             Q = q + self.L * sigma
        #             R = r + self.L * tau
        #             S = s + self.L * tau
        #             # use only upper left (up-up) block of G
        #             H_FO += openfermion.FermionOperator(FO_string.format(P=P, Q=Q, R=R, S=S), G[p, q, r, s])
        # elif self.spin_mode == "1/2":
        if True:
            i = 0
            for P, R in itertools.combinations_with_replacement(range(2 * self.L), 2):  # sum over P < R
                if P == R: continue
                j = 0
                for Q, S in itertools.combinations_with_replacement(range(2 * self.L), 2):  # sum over Q < S
                    if Q == S: continue
                    coeff_PQRS = F[i, j]  # = F_{(PR)}{(QS)}
                    H_FO += openfermion.FermionOperator(FO_string.format(P=P, Q=Q, R=R, S=S), coeff_PQRS)
                    j += 1
                i += 1
            # else:
            #     for P, Q, R, S in itertools.product(range(2 * self.L), repeat=4):
            #         H_FO += openfermion.FermionOperator(FO_string.format(P=P, Q=Q, R=R, S=S), G[P, Q, R, S])

        if print_symbolic_op:
            print(H_FO)

        return self.convert_FO_to_csc(H_FO)

    def build_S_z(self):  # S_z spin operator
        S_z_FO = openfermion.FermionOperator("{maxorb}^ {maxorb}".format(maxorb=self.LL - 1), 0)
        
        # S_z is inviariant under spin symmetry breaking (yet spin nonmixing) orbital rotations 
        for m in range(self.L):
            S_z_FO += openfermion.FermionOperator("{m_up}^ {m_up}".format(m_up=m), 0.5) \
                      - openfermion.FermionOperator("{m_down}^ {m_down}".format(m_down=m + self.L), 0.5)

        return self.convert_FO_to_csc(S_z_FO)

    def build_S_plus(self):  # S_plus spin operator
        S_plus_FO = openfermion.FermionOperator("{maxorb}^ {maxorb}".format(maxorb=self.LL - 1), 0)
        
        # S_+ is not invariant under spin symmetry breaking orbital rotations:
        # S_+ = \Sum_{p\mu\nu} C^{\uparrow}_{p\mu} C^{\downarrow}_{p\nu} \a^\dagger_{\mu\uparrow} \a^_{\nu\downarrow}

        coeff = np.einsum("pm, pn -> mn", self.C_sbd[:self.L, :self.L], self.C_sbd[self.L:, self.L:])

        for m, n in itertools.product(range(self.L), repeat=2):
            S_plus_FO += openfermion.FermionOperator("{m_up}^ {n_down}".format(m_up=m, n_down=n + self.L), coeff[m, n])

        return self.convert_FO_to_csc(S_plus_FO)

    def build_S_minus(self):  # S_minus spin operator
        S_minus_FO = openfermion.FermionOperator("{maxorb}^ {maxorb}".format(maxorb=self.LL - 1), 0)

        # S_- is not invariant under spin symmetry breaking orbital rotations
        # S_- = \Sum_{p\mu\nu} C^{\downarrow}_{p\mu} C^{\uparrow}_{p\nu} \a^\dagger_{\mu\downarrow} \a^_{\nu\uparrow}

        coeff = np.einsum("pm, pn -> mn", self.C_sbd[self.L:, self.L:], self.C_sbd[:self.L, :self.L])

        for m, n in itertools.product(range(self.L), repeat=2):
            S_minus_FO += openfermion.FermionOperator("{m_down}^ {n_up}".format(m_down=m + self.L, n_up=n), coeff[m, n])

        return self.convert_FO_to_csc(S_minus_FO)

    def build_S_squared(self):  # S^2 spin operator
        S_minus = self.build_S_minus()
        S_plus = self.build_S_plus()
        S_z = self.build_S_z()

        S_squared = S_minus @ S_plus + S_z @ S_z + S_z

        return S_squared

    def build_F(self, G, argsort=None):  # This builds an antisymmetric F: F_{(PR)(QS)} = 2(G_{PQRS} - G_{PSRQ})
        LL = G.shape[0]

        F = np.zeros([int(LL * (LL - 1) / 2), int(LL * (LL - 1) / 2)])

        # This loop is not very pythonic and should be improved
        i = 0
        for P, R in itertools.combinations_with_replacement(range(LL), 2):  # sum over P < R
            if P == R: continue
            j = 0
            for Q, S in itertools.combinations_with_replacement(range(LL), 2):  # sum over Q < S
                if Q == S: continue
                F[i, j] = 0
                F[i, j] = 2 * (G[P, Q, R, S] - G[P, S, R, Q])
                j += 1
            i += 1

        return F

    def build_F_k(self, l, U, k):  # F_k = lambda_k * U_k^{(PR)} * U_k^{(QS)}
        index_mask = np.zeros(len(l))
        index_mask[k] = 1
        l_k = l * index_mask

        F_k = U @ np.diag(l_k) @ U.conj().T

        return F_k
