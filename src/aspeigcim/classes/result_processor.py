import numpy as np
import matplotlib.pyplot as plt
import os


class ResultProcessor:
    def __init__(self, config):
        self.evolution_result  = config["evolution_result"]
        self.result_name       = config["result_name"]
        self.T                 = config["T"]
        self.ds                = config["ds"]
        self.num_eig_plotted   = config["num_eig_plotted"]
        # self.atoms_config_path = config["atoms_config_path"]
        self.result_dir        = config["result_dir"]
        self.plot_energies     = config["plot_energies"]
        self.plot_gaps         = config["plot_gaps"]
        self.plot_S_sq         = config["plot_S_sq"]
        self.plot_S_z          = config["plot_S_z"]
        self.plot_final_gs_overlap    = config["plot_gs_overlap"]
        self.plot_initial_gs_overlap  = config["plot_init_overlap"]
        self.plot_atoms        = config["plot_atoms"]
        self.use_physical_time = config["use_physical_time"]

        self.energies = self.evolution_result["energies"]
        self.final_ground_state_overlaps = self.evolution_result["final_ground_state_overlaps"]
        self.initial_ground_state_overlaps = self.evolution_result["initial_ground_state_overlaps"]
        self.S_sq_evs = self.evolution_result["S_sq_evs"]
        self.S_z_evs = self.evolution_result["S_z_evs"]
        self.t = self.evolution_result["t"]

        self.file_dir = os.path.dirname(os.path.realpath(__file__))

    def save_walk_result(self):
        np.savetxt("{}/result_{}.txt".format(self.result_dir, self.result_name),
                   self.evolution_result[0])

    def plot_evolution_result(self, log_y_axis=False):
        energies = self.energies
        S_sq_evs = self.S_sq_evs
        S_z_evs = self.S_z_evs
        final_ground_state_overlaps = self.final_ground_state_overlaps
        initial_ground_state_overlaps = self.initial_ground_state_overlaps
        n_eig = self.num_eig_plotted
        if(n_eig == -1 or n_eig > energies.shape[1]):
            n_eig = energies.shape[1]
        x_label = "t" if self.use_physical_time else "s"

        plt.figure(figsize=(12, 8))

        # self.walk_plot_title = "{}_{}{}".format(self.atoms_config_name, self.basis_set, self.plot_label)
        plt.suptitle(self.result_name)
        plt.rcParams["figure.figsize"] = (200, 300)

        x = self.t if self.use_physical_time else np.linspace(0, 1, int(1 / self.ds) + 1)

        subplot_index = 0
        n_plots = self.plot_energies + self.plot_gaps + self.plot_S_sq + self.plot_S_z + self.plot_final_gs_overlap\
                  + self.plot_initial_gs_overlap + self.plot_atoms
        n_subplot_rows = (n_plots + 1) // 2
        n_subplot_cols = 1 if n_plots == 1 else 2

        if self.plot_energies:
            subplot_index += 1
            plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
            # plt.xlabel("eigenvalues of F \"turned on\"")
            plt.xlabel(x_label)
            plt.ylabel("energy (a.u.)")
            for i in range(n_eig):
                plt.plot(x, energies[:, i], label="e_{}".format(i))
            # plt.legend(loc="upper right")

        if self.plot_gaps:
            subplot_index += 1
            plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
            # plt.xlabel("eigenvalues of F \"turned on\"")
            plt.xlabel(x_label)
            plt.ylabel("energy gap (a.u.)")
            plt.yscale("log")
            for i in range(n_eig - 1):
                gap_i = energies[:, i + 1] - energies[:, i]
                gap_i[np.isclose(gap_i, 0, atol=1e-14)] = 1e-15
                plt.plot(x, gap_i, label="e_{} - e_{}".format(i + 1, i))
            plt.legend(loc="upper right")

        if self.plot_S_sq:
            subplot_index += 1
            plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
            # plt.xlabel("eigenvalues of F \"turned on\"")
            plt.xlabel(x_label)
            plt.ylabel("<S^2>")
            for i in range(n_eig):
                plt.plot(x, S_sq_evs[:, i], label="<S^2>_{}".format(i))
            plt.legend(loc="lower left")

        if self.plot_S_z:
            subplot_index += 1
            plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
            # plt.xlabel("eigenvalues of F \"turned on\"")
            plt.xlabel(x_label)
            plt.ylabel("<Sz>")
            for i in range(n_eig):
                plt.plot(x, S_z_evs[:, i], label="<Sz>_{}".format(i))
            plt.legend(loc="lower left")

        if self.plot_final_gs_overlap:
            subplot_index += 1
            plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
            # plt.xlabel("eigenvalues of F \"turned on\"")
            plt.xlabel(x_label)
            plt.ylabel("final ground state overlap")
            if log_y_axis: plt.yscale("log")
            for i in range(n_eig):
                plt.plot(x, final_ground_state_overlaps[:, i], label="gs_f_ovlp_{}".format(i))
            plt.legend(loc="lower right")

        if self.plot_initial_gs_overlap:
            subplot_index += 1
            plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
            plt.xlabel(x_label)
            plt.ylabel("initial ground state overlap")
            if log_y_axis: plt.yscale("log")
            for i in range(n_eig):
                plt.plot(x, initial_ground_state_overlaps[:, i], label="gs_i_ovlp_{}".format(i))
            plt.legend(loc="lower right")

        if self.plot_atoms:
            subplot_index += 1
            plt.subplot(n_subplot_rows, n_subplot_cols, subplot_index)
            atoms_config_array = np.loadtxt(self.atoms_config_path)
            plt.scatter(x=atoms_config_array[:, 1], y=atoms_config_array[:, 2])
            plt.xlabel("x (Å)")
            plt.ylabel("y (Å)")
            plt.xlim([-11, 11])
            plt.ylim([-11, 11])
            plt.gca().set_aspect('equal', adjustable='box')

        self.walk_plot_fig = plt.gcf()
        plt.show()

    def save_walk_result_plot(self):
        self.walk_plot_fig.savefig("{}/{}.pdf".format(self.result_dir, self.result_name))

    def find_success_times(self):
        first_occurrence_75 = np.argmax(self.final_ground_state_overlaps[:, 0] >= 0.75) * self.T * self.ds
        first_occurrence_90 = np.argmax(self.final_ground_state_overlaps[:, 0] >= 0.90) * self.T * self.ds
        first_occurrence_99 = np.argmax(self.final_ground_state_overlaps[:, 0] >= 0.99) * self.T * self.ds
        max_overlap = np.max(self.final_ground_state_overlaps[:, 0])
        final_overlap = self.final_ground_state_overlaps[-1, 0]

        print("first_occurrence_75 = {}".format(first_occurrence_75))
        print("first_occurrence_90 = {}".format(first_occurrence_90))
        print("first_occurrence_99 = {}".format(first_occurrence_99))
        print("max_overlap = {}".format(max_overlap))
        print("final_overlap = {}".format(final_overlap))