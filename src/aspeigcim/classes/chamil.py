import ctypes
import os


class CHamil(object):
    def __init__(self, k, N, spin_modes=1):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL("{}/../lib/libchamil.so".format(file_dir))

        self.lib.CHamil_new.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.CHamil_new.restype = ctypes.c_void_p

        self.lib.CHamil_populateHamiltonian.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

        self.lib.CHamil_show4DTensor.argtypes = [ctypes.c_void_p]
        self.lib.CHamil_show4DTensor.restype = ctypes.c_void_p

        self.obj = self.lib.CHamil_new(k, N, spin_modes)

    def populate_hamiltonian(self, H, G):
        H_p = H.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        G_p = G.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.lib.CHamil_populateHamiltonian(self.obj, H_p, G_p)

    def show_4D_tensor(self, T):
        T_p = T.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.lib.CHamil_show4DTensor(self.obj, T_p)
