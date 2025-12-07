def scf_manual(self, h, g, S=None):
    # return np.array([[1, -1], [1, 1]]) / np.sqrt(2)

    L = self.L
    sm = self.spin_levels
    if S is None:
        S = np.identity(L)

    # np.random.seed(0)
    # P = np.random.randn(k, k)  # density matrix
    # P = np.zeros((k, k))
    P = np.identity(L)
    P = (P + P.T.conj()) / 2
    P_prev = P + np.inf

    C = np.zeros(L)  # orbital rotation matrix
    e = 0  # energy
    e_prev = np.inf

    while np.linalg.norm(P - P_prev) > 1e-14 and np.abs(e - e_prev) > 1e-14:
        f = h + 0.5 * (sm * np.einsum("kl, mnkl -> mn", P, g) - np.einsum("kl, mknl -> mn", P, g))

        _, C = scipy.linalg.eigh(f, S)

        C_occ = C[:, :self.N // sm]
        P_prev = P
        P = 2 * np.matmul(C_occ, C_occ.T)

        e_prev = e
        e = sm * 0.25 * np.einsum("mn, mn ->", P, h + f)  # + E_nuc
        print("e = {}".format(e))

    # normalisation
    orb_norms = np.diag(C.T @ S @ C)  # norms of unnormalised MOs
    C = C / np.sqrt(orb_norms)

    print("C_hubb = \n{}".format(C))

    return C
