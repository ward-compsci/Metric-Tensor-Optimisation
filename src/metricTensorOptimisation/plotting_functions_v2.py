import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from matplotlib import rcParams

import networkx as nx
import pickle

from scipy.interpolate import griddata

from collections import defaultdict

# Use Times New Roman font
rcParams['font.family'] = 'Times New Roman'


class PlottingFunctions:
    def __init__(self, parent):
        self.parent = parent

    def _load_from_pickle(self, filepath):
        if os.path.exists(filepath):
            print(f"Loading results for {filepath}")
            with open(filepath, "rb") as f:
                return pickle.load(f)
            raise FileNotFoundError(f"File not found: {filepath}")
        
    def _find_nearest_index(self, array, value):
        return np.abs(array - value).argmin()
    
    def _get_signals(self, filepath):
        data = self._load_from_pickle(filepath)
        return data["Reconstructed signals"]

    def _get_particle_data(self, filepath):
        data = self._load_from_pickle(filepath)
        particle_history = data["Particle history"]
        error_history = data["Error history"]
        
        best_particle_idx = np.argmin(error_history[-1])
        best_particle = particle_history[-1][best_particle_idx]
        
        print(f"Best particle: {best_particle}")
        print(f"Error {error_history[-1][best_particle_idx]}")
        print(f"Function mapping: \n{self.parent.graph_instance.num2func_dict}")
        print(f"Best particle combination: {self.parent.graph_instance.num2node_dict[int(best_particle[0])]}")


        return best_particle, best_particle_idx
    
    def _get_stim_average_blocks(self, signals):
        """
        Proxy function that takes a signal and returns the stim averaged blocks
        It will take a signal of the form [time, channels, wavelengths]
        """

        T, Ch, W, P = np.shape(signals)

        # Stats meta-data -> defined from parent class
        # If using different optimisation hyper-parameters this will need to change
        dur = 0.5; peak = 5.9; post_start = peak - (dur / 2); post_start2 = peak * 2 - (dur / 2)

        stim_time = np.linspace(-5, 15, int((5 + 15) * self.parent.fs))
        stim_time_num = len(stim_time)
        t0_index = self._find_nearest_index(stim_time, 0)

        self.stats_meta = [stim_time, stim_time_num, dur, peak, post_start, post_start2]

        mean_blocks = np.zeros([stim_time_num, Ch, W, P]); std_blocks = np.zeros([stim_time_num, Ch, W, P])

        for p in range(P):
            blocks = np.zeros([stim_time_num, Ch, W, self.parent.num_stim_events])
            for i, stimulus in enumerate(self.parent.stim_data[p,:]):
                stim_idx = self._find_nearest_index(self.parent.time, stimulus - 5)
                stim_block = signals[np.ix_(
                                     np.arange(stim_idx, stim_idx+stim_time_num),
                                     np.arange(Ch),
                                     np.arange(2),
                                     [p]
                                     )]
                stim_block = stim_block.squeeze(-1)
                blocks[:, :, :, i] = stim_block - stim_block[t0_index, :, :]

            mean_blocks[:, :, :, p] = np.mean(blocks, axis=-1); std_blocks[:, :, :, p] = np.std(blocks, axis=-1)

        return mean_blocks, std_blocks

    def print_particles_data(self, filepath):

        data = self._load_from_pickle(filepath)
        particle_history = data["Particle history"]
        error_history = data["Error history"]
        
        print(particle_history[-1])
        print(error_history[-1])
    
    # -------------------------
    # Plotting helper functions
    # -------------------------

    def _plot_raw_data(self, axes, ch, ch_idx, col, p):
        axes[ch,col].plot(self.parent.time, self.parent.snirf_data[:,ch_idx[ch],0,p], label=f"{self.parent.wavelength_labels[0]:.0f}nm",
                          linestyle='--', linewidth=0.5)
        axes[ch,col].plot(self.parent.time, self.parent.snirf_data[:,ch_idx[ch],1,p], label=f"{self.parent.wavelength_labels[1]:.0f}nm",
                          linestyle='--', linewidth=0.5)

    def _plot_mean_signal_blocks(self, axes, ch, col, p, stim_time, mean_blocks):
        axes[ch,col].plot(stim_time, mean_blocks[:,ch,0,p], color='r', label='O2Hb', linewidth=2)
        axes[ch,col].plot(stim_time, mean_blocks[:,ch,1,p], color='b', label='HHb', linewidth=2)

    def _add_mean_block_signal_metadata(self, axes, ch, col):

        stim_time, stim_time_num, dur, peak, post_start, post_start2 = self.stats_meta

        try:
            axes[ch,col].axvspan(0,2,color='purple',alpha=0.3)
            
            axes[ch,col].axvline(peak, color='purple', linestyle='--', alpha=0.7)
            axes[ch,col].axvline(peak*2, color='purple', linestyle='--', alpha=0.7)
            axes[ch,col].axvline(0 - dur / 2, color='purple', linestyle='--', alpha=0.7)

            
            axes[ch,col].axvspan(0 - dur, 0, color='grey', alpha=0.1)
            axes[ch,col].axvspan(post_start, post_start+ dur,color='grey',alpha=0.1)
            axes[ch,col].axvspan(post_start2, post_start2 + dur,color='grey',alpha=0.1)
        except:
            axes.axvspan(0,2,color='purple',alpha=0.3)
            
            axes.axvline(peak, color='purple', linestyle='--', alpha=0.7)
            axes.axvline(peak*2, color='purple', linestyle='--', alpha=0.7)
            axes.axvline(0 - dur / 2, color='purple', linestyle='--', alpha=0.7)
            
            axes.axvspan(0 - dur, 0, color='grey', alpha=0.1)
            axes.axvspan(post_start, post_start+ dur,color='grey',alpha=0.1)
            axes.axvspan(post_start2, post_start2 + dur,color='grey',alpha=0.1)


    def _add_stim_times(self, axes, ch, col, p):
        for s in range(self.parent.num_stim_events):
            axes[ch,0].axvspan(self.parent.stim_data[p, s], self.parent.stim_data[p, s]+2, color='purple', alpha=0.3)

    def _plot_mean_signal_blocks_std(self, axes, ch, col, p, stim_time, mean_blocks, std_blocks):
        axes.plot(stim_time, np.mean(mean_blocks[:, :, 0, p], axis=1), color='r', label="HbO")
        axes.plot(stim_time, np.mean(mean_blocks[:, :, 1, p], axis=1), color='b', label="HbR")

        axes.fill_between(stim_time, np.mean(mean_blocks[:, :, 0, p], axis=1) - np.std(mean_blocks[:, :, 0, p], axis=1),
                                   np.mean(mean_blocks[:, :, 0, p], axis=1) + np.std(mean_blocks[:, :, 0, p], axis=1), alpha=0.3, color='r')
        axes.fill_between(stim_time, np.mean(mean_blocks[:, :, 1, p], axis=1) - np.std(mean_blocks[:, :, 1, p], axis=1),
                            np.mean(mean_blocks[:, :, 1, p], axis=1) + np.std(mean_blocks[:, :, 1, p], axis=1), alpha=0.3, color='b')


    def _configure_axis(self, axes, ylabel=None, xlabel=None, ylims=None):
        try:
            for ax in axes.flatten():
                ax.legend(fontsize=10)
                ax.set_xlabel(xlabel)
                ax.grid()
        except:
            axes.legend(fontsize=10)
            axes.set_xlabel(xlabel)
            axes.grid()
            axes.set_ylabel(ylabel)

    def _set_small_fig(self, fig, axes, ylims=None):

        fig.set_size_inches(3,2)

        try:
            for ax in axes.flatten():
                ax.tick_params(axis="both", which="major", labelsize=10)
                ax.tick_params(axis="both", which="minor", labelsize=6)
                ax.set_ylim(ylims)
        except:
                axes.tick_params(axis="both", which="major", labelsize=10)
                axes.tick_params(axis="both", which="minor", labelsize=6)
                axes.set_ylim(ylims)

    def _set_large_fig(self, fig, axes, ylims=None):

        fig.set_size_inches(7,4)

        try:
            for ax in axes.flatten():
                ax.tick_params(axis="both", which="major", labelsize=10)
                ax.tick_params(axis="both", which="minor", labelsize=6)
                ax.set_ylim(ylims)
        except:
                axes.tick_params(axis="both", which="major", labelsize=10)
                axes.tick_params(axis="both", which="minor", labelsize=6)
                axes.set_ylim(ylims)
    # -------------------------
    # Plotting functions
    # -------------------------

    def plot_optimisation_results(self, filepath, channels=None):

        ch_idx = np.arange(self.parent.num_channels) if channels is None else np.array(channels) - 1
        
        ch_idx = self.parent.ROI_channels_index_combined
        num_channels = len(ch_idx)

        best_particle, best_particle_index = self._get_particle_data(filepath)
        all_particle_signals = self._get_signals(filepath)

        signals_best_particle = all_particle_signals[best_particle_index]

        _ = self.parent.is_active(signals_best_particle, print_result=True)

        mean_blocks, _ = self._get_stim_average_blocks(signals_best_particle)
        stim_time, stim_time_num, dur, peak, post_start, post_start2 = self.stats_meta

        for p in range(self.parent.num_participants):
            fig, axes = plt.subplots(num_channels, 2, constrained_layout=True, sharey='col')

            for ch in range(num_channels):
                self._plot_raw_data(axes=axes,ch=ch,ch_idx=ch_idx,col=0,p=p)
                self._add_stim_times(axes=axes, ch=ch, col=0, p=p)

                self._plot_mean_signal_blocks(axes=axes, ch=ch, col=1, p=p, stim_time=stim_time, mean_blocks=mean_blocks)
                self._add_mean_block_signal_metadata(axes, ch, 1)

            self._configure_axis(axes)
            self._set_large_fig(fig, axes)

        plt.show()

    def plot_optimisation_results_mean_channel(self, filepath, channels=None, print_plots=False, ylims=None):

        ch_idx = np.arange(self.parent.num_channels) if channels is None else np.array(channels) - 1

        ch_idx = self.parent.ROI_channels_index_combined

        num_channels = len(ch_idx)

        best_particle, best_particle_index = self._get_particle_data(filepath)
        all_particle_signals = self._get_signals(filepath)

        signals_best_particle = all_particle_signals[best_particle_index]

        _ = self.parent.is_active(signals_best_particle, print_result=True)

        mean_blocks, std_blocks = self._get_stim_average_blocks(signals_best_particle)# ; print(f"\n====Pre-shape: {mean_blocks.shape}\n====")
        mean_blocks, std_blocks = mean_blocks[:,ch_idx,:,:], std_blocks[:,ch_idx,:,:]# ; print(f"\n====Post shape: {mean_blocks.shape}\n====")
        stim_time, stim_time_num, dur, peak, post_start, post_start2 = self.stats_meta

        if print_plots != False:

            for p in range(self.parent.num_participants):
                fig, axes = plt.subplots(1, 1, layout="constrained")

                self._plot_mean_signal_blocks_std(axes=axes, ch=1, col=1, p=p, stim_time=stim_time, mean_blocks=mean_blocks, std_blocks=std_blocks)
                self._add_mean_block_signal_metadata(axes, ch=1, col=1)

                self._configure_axis(axes, ylabel="Concentration (μM)")
                self._set_small_fig(fig, axes)
                
                fig.savefig(f"reconstructions{p+1}.pdf")
                
                fig, axes = plt.subplots(1, 1, layout="constrained")

                for i in range(len(channels)):
                    if i == 0:
                        axes.plot(self.parent.time, self.parent.snirf_data[:,ch_idx[i],0,p], label=f"{self.parent.wavelength_labels[0]:.0f}nm",
                                linestyle='--', linewidth=0.5, color='tab:orange')
                        axes.plot(self.parent.time, self.parent.snirf_data[:,ch_idx[i],1,p], label=f"{self.parent.wavelength_labels[1]:.0f}nm",
                                linestyle='--', linewidth=0.5, color='tab:blue')
                    else:
                        axes.plot(self.parent.time, self.parent.snirf_data[:,ch_idx[i],0,p], linestyle='--', linewidth=0.5,  color='tab:orange')
                        axes.plot(self.parent.time, self.parent.snirf_data[:,ch_idx[i],1,p], linestyle='--', linewidth=0.5, color='tab:blue')
                    
                
                for s in range(self.parent.num_stim_events):
                    axes.axvspan(self.parent.stim_data[p, s], self.parent.stim_data[p, s]+2, color='purple', alpha=0.3)


                self._configure_axis(axes, ylabel="Intensity (A.U.)")
                self._set_small_fig(fig, axes, ylims=(0.0, 0.6))

                fig.savefig(f"raw_data{p+1}.pdf")

            plt.show()

    
    def plot_optimisation_results_mean_channel_synthetic(self, filepath, channels=None):

        ch_idx = np.arange(self.parent.num_channels) if channels is None else np.array(channels) - 1
        num_channels = len(ch_idx)

        best_particle, best_particle_index = self._get_particle_data(filepath)
        all_particle_signals = self._get_signals(filepath)

        signals_best_particle = all_particle_signals[best_particle_index]

        _ = self.parent.is_active(signals_best_particle, print_result=True)

        mean_blocks, std_blocks = self._get_stim_average_blocks(signals_best_particle)
        stim_time, stim_time_num, dur, peak, post_start, post_start2 = self.stats_meta

        for p in range(self.parent.num_participants):
            fig, axes = plt.subplots(1, 1, layout="constrained")

            self._plot_mean_signal_blocks_std(axes=axes, ch=1, col=1, p=p, stim_time=stim_time, mean_blocks=mean_blocks, std_blocks=std_blocks)
            self._add_mean_block_signal_metadata(axes, ch=1, col=1)

            self._configure_axis(axes, ylabel="Concentration (μM)")
            self._set_small_fig(fig, axes)
            
            fig.savefig(f"reconstructions{p}.pdf")
            
            fig, axes = plt.subplots(1, 1, layout="constrained")

            offset = 3
            for i in range(3):
                # Extract signals
                y1 = self.parent.snirf_data[:, ch_idx[i], 0, p]
                y2 = self.parent.snirf_data[:, ch_idx[i], 1, p]

                # Normalise by subtracting median or mean to center
                y1_centered = y1 - np.median(y1) + i * offset
                y2_centered = y2 - np.median(y2) + i * offset                             

                if i == 0:
                    axes.plot(self.parent.time, y1_centered, label=f"{self.parent.wavelength_labels[0]:.0f}nm",
                            linestyle='--', linewidth=0.5, color='tab:orange')
                    axes.plot(self.parent.time, y2_centered, label=f"{self.parent.wavelength_labels[1]:.0f}nm",
                            linestyle='--', linewidth=0.5, color='tab:blue')
                else:
                    axes.plot(self.parent.time, y1_centered, linestyle='--', linewidth=0.5,  color='tab:orange')
                    axes.plot(self.parent.time, y2_centered, linestyle='--', linewidth=0.5, color='tab:blue')

                for s in range(self.parent.num_stim_events):
                    axes.axvspan(self.parent.stim_data[p, s], self.parent.stim_data[p, s]+2, color='purple', alpha=0.3)

                self._configure_axis(axes, ylabel="Intensity (A.U.)")
                self._set_small_fig(fig, axes, ylims=[-2, 8])

                fig.savefig(f"raw_data{p}.pdf")

        plt.show()

    def plot_param_history(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")
        with open(filepath, "rb") as f:
            file = pickle.load(f)
        
        particle_history = file["Particle history"]
        params_history = []

        for step, particles in enumerate(particle_history):
            params_history.append(particles[:,1:])

        print(np.array(params_history).shape)

        params_history = np.asarray(params_history)
        params_history = params_history[:,0:10,:]

        steps, n_particles, _ = params_history.shape

        fig, ax = plt.subplots()
        scat = ax.scatter([], [], s=50)

        # For trails, store previous positions (max length of trail)
        trail_length = steps  # number of past frames to keep visible

        # Initialize a list of particle trails: one per particle
        # Each trail is a list of positions over last 'trail_length' steps
        particle_trails = [np.empty((0, 2)) for _ in range(n_particles)]

        ax.set_xlim(np.min(params_history[:,:,0]) - 0.1, np.max(params_history[:,:,0]) + 0.1)
        ax.set_ylim(np.min(params_history[:,:,1]) - 0.1, np.max(params_history[:,:,1]) + 0.1)
        ax.set_title("Particle trajectories (2 continuous dimensions)")
        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")

        def update(frame):
            ax.clear()
            ax.set_xlim(np.min(params_history[:,:,0]) - 0.1, np.max(params_history[:,:,0]) + 0.1)
            ax.set_ylim(np.min(params_history[:,:,1]) - 0.01, np.max(params_history[:,:,1]) + 0.01)
            ax.set_title("Particle trajectories (2 continuous dimensions)")
            ax.set_xlabel("Parameter 1")
            ax.set_ylabel("Parameter 2")

            ax.set_title(f"Step {frame+1}")

            # Update trails for each particle
            for i in range(n_particles):
                # Append current position to trail
                if len(particle_trails[i]) >= trail_length:
                    particle_trails[i] = np.vstack([particle_trails[i][1:], params_history[frame, i]])
                else:
                    particle_trails[i] = np.vstack([particle_trails[i], params_history[frame, i]])

                # Plot trail with fading alpha
                decay_factor = 3

                for t_idx, pos in enumerate(particle_trails[i]):
                    alpha = 0.8 * np.exp(-decay_factor * (len(particle_trails[i]) - t_idx - 1) / len(particle_trails[i]))
                    ax.plot(pos[0], pos[1], 'o', color=f"C{i % 10}", alpha=alpha)

            return []

        anim = FuncAnimation(fig, update, frames=range(steps), interval=500, blit=False)

        plt.show()


    def plot_node_map(self, nodes, errors, adjacency_matrix):

        with open(adjacency_matrix, "rb") as f:
            adj, node2num_dict, num2node_dict = pickle.load(f)


        def reemovNestings(l, output):
            for i in l:
                if type(i) == list:
                    reemovNestings(i, output)
                else:
                    output.append(i)

        node_error_dict = {}
        node_count_dict = {}

        for n in range(len(nodes)):    
            for solution_path, error_path in zip(nodes[n], errors[n]):
                for node_array, err_vec in zip(solution_path, error_path):
                    node_id = (node_array[0])
                    if node_id in node_error_dict:
                        node_error_dict[node_id] += err_vec
                        node_count_dict[node_id] += 1
                    else:
                        node_error_dict[node_id] = np.array(err_vec, dtype=float)
                        node_count_dict[node_id] = 1

        print(node_error_dict[102])

        # Normalize by count for each node
        for node_id in node_error_dict:
            node_error_dict[node_id] /= node_count_dict[node_id]
        
        print(node_error_dict[102])

        # 1. Convert all to float arrays (in case some are still int)
        for k in node_error_dict:
            node_error_dict[k] = node_error_dict[k].astype(float)

        # 2. Normalize all errors to [0, 1]
        all_errors = np.array([v for v in node_error_dict.values()])
        min_err = all_errors.min()
        max_err = all_errors.max()

        for k in node_error_dict:
            node_error_dict[k] = (node_error_dict[k] - min_err) / (max_err - min_err + 1e-12)

        # 3. Invert so that lower error becomes higher peak
        for k in node_error_dict:
            node_error_dict[k] = 1.0 - node_error_dict[k]

        print(node_error_dict[102])

        N = adj.shape[0]
        G = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)

        # 1) Get 2D layout of nodes
        pos = nx.spring_layout(G, seed=42)  # dict: node -> (x,y)

        from collections import defaultdict

        def hasse_diagram(combo_dict, x_spacing=1.0, y_base_spacing=1.0):
            # Group combos by length
            length_groups = defaultdict(list)
            for combo, idx in combo_dict.items():
                length_groups[len(combo)].append((idx, combo))

            pos = {}
            sorted_lengths = sorted(length_groups)

            scale_fn = lambda l: l ** 2

            # Map each unique length to a scaled y-position
            y_positions = {}
            current_y = 0
            for l in sorted_lengths:
                y_positions[l] = current_y
                current_y += y_base_spacing * scale_fn(l)

            for l in sorted_lengths:
                group = sorted(length_groups[l], key=lambda x: x[0])  # Optional: sort by index
                n = len(group)
                for i, (idx, _) in enumerate(group):
                    x = (i - (n - 1) / 2) * x_spacing
                    y = y_positions[l]
                    pos[idx] = (x, y)

            return pos

        pos = hasse_diagram(node2num_dict)

        # Extract x, y, and error values aligned with node indices
        xs = np.array([pos[i][0] for i in range(N)])
        ys = np.array([pos[i][1] for i in range(N)])
        zs = np.array([node_error_dict.get(i, 0) for i in range(N)])  # height = error

        # 2) Create grid for interpolation
        grid_x, grid_y = np.mgrid[xs.min():xs.max():200j, ys.min():ys.max():200j]

        # 3) Interpolate error values over grid
        grid_z = griddata((xs, ys), zs, (grid_x, grid_y), method='linear', fill_value=0)
        grid_z = np.clip(grid_z, 0, None)
        

        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111, projection='3d')

        # Plot only nodes that have error values
        for x, y, z in zip(xs, ys, zs):
            if z > 0:  # or if z is not None
                ax.bar3d(x, y, 0, 1, 1, z, shade=True)

        # Optionally plot the nodes without height as small dots on base plane
        for x, y, z in zip(xs, ys, zs):
            if z == 0:
                ax.scatter(x, y, 0, color='gray', s=10)

        
        # Add labels on nodes (optional)
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            if z > .81:
                ax.text(x+0.1, y, z+0.1, str(i), color='black', fontsize=8, ha='center', va='center')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=6)

        ax.set_xlabel('X (A.U.)')
        ax.set_ylabel('Y (A.U.)')
        ax.set_zlabel('Inverted normalised error')

        plt.savefig(f"surface.pdf")

        plt.show()


    def plot_node_map_topdown(self, nodes, errors, adjacency_matrix):
        """
        Create a top-down projection of nodes where:
        - Circle size indicates how many times a node appears
        - Circle color represents the error (inverted and normalized)
        - Distance between nodes is based on maximum bounding circle radius
        """
        
        with open(adjacency_matrix, "rb") as f:
            adj, node2num_dict, num2node_dict = pickle.load(f)

        node_error_dict = {}
        node_count_dict = {}

        # Collect error data and counts for each node
        for n in range(len(nodes)):    
            for solution_path, error_path in zip(nodes[n], errors[n]):
                for node_array, err_vec in zip(solution_path, error_path):
                    node_id = (node_array[0])
                    if node_id in node_error_dict:
                        node_error_dict[node_id] += err_vec
                        node_count_dict[node_id] += 1
                    else:
                        node_error_dict[node_id] = np.array(err_vec, dtype=float)
                        node_count_dict[node_id] = 1

        # Normalize errors by count
        for node_id in node_error_dict:
            node_error_dict[node_id] /= node_count_dict[node_id]

        # Convert to float arrays and normalize errors to [0, 1]
        for k in node_error_dict:
            node_error_dict[k] = node_error_dict[k].astype(float)

        all_errors = np.array([v for v in node_error_dict.values()])
        min_err = all_errors.min()
        max_err = all_errors.max()

        for k in node_error_dict:
            node_error_dict[k] = (node_error_dict[k] - min_err) / (max_err - min_err + 1e-12)

        # Invert so that lower error becomes higher value (better performance = darker colors)
        for k in node_error_dict:
            node_error_dict[k] = 1.0 - node_error_dict[k]

        # Create network graph and get layout
        N = adj.shape[0]
        G = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)

        from collections import defaultdict

        def hasse_diagram_concentric(combo_dict):
            # Group combos by length (which determines the radius)
            length_groups = defaultdict(list)
            for combo, idx in combo_dict.items():
                length_groups[len(combo)].append((idx, combo))

            pos = {}
            sorted_lengths = sorted(length_groups)
            
            for length in sorted_lengths:
                group = sorted(length_groups[length], key=lambda x: x[0])
                n = len(group)
                radius = length + 1  # Empty set at radius 1, first combo at radius 2, etc.
                
                if n == 1:
                    # Single node at center for empty set or single item
                    if length == 0:
                        pos[group[0][0]] = (0, 0)
                    else:
                        pos[group[0][0]] = (radius, 0)
                else:
                    # Distribute nodes evenly around the circle
                    for i, (idx, _) in enumerate(group):
                        angle = 2 * np.pi * i / n
                        x = radius * np.cos(angle)
                        y = radius * np.sin(angle)
                        pos[idx] = (x, y)

            return pos

        pos = hasse_diagram_concentric(node2num_dict)

        # Calculate circle sizes based on node counts
        max_count = max(node_count_dict.values()) if node_count_dict else 1
        min_count = min(node_count_dict.values()) if node_count_dict else 1
        
        # Normalize circle sizes (min_size to max_size)
        min_circle_size = 50
        max_circle_size = 500
        
        circle_sizes = {}
        for node_id in range(N):
            if node_id in node_count_dict:
                normalized_count = (node_count_dict[node_id] - min_count) / (max_count - min_count + 1e-12)
                circle_sizes[node_id] = min_circle_size + normalized_count * (max_circle_size - min_circle_size)
            else:
                circle_sizes[node_id] = min_circle_size

        # Calculate maximum radius for spacing
        max_radius = np.sqrt(max_circle_size / np.pi)
        
        # Adjust positions to ensure minimum distance between circles
        adjusted_pos = {}
        for node_id in range(N):
            if node_id in pos:
                x, y = pos[node_id]
                # Scale positions to ensure circles don't overlap
                adjusted_pos[node_id] = (x * max_radius * 2.5, y * max_radius * 2.5)
            else:
                adjusted_pos[node_id] = (0, 0)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot circles for each node
        for node_id in range(N):
            if node_id in node_error_dict and node_error_dict[node_id] > 0:
                x, y = adjusted_pos[node_id]
                size = circle_sizes[node_id]
                error_value = node_error_dict[node_id]
                radius = np.sqrt(size / np.pi)
                
                # Use blue colormap (light blue = low performance, dark blue = high performance)
                circle = plt.Circle((x, y), radius, 
                                  color=plt.cm.Blues(error_value), 
                                  alpha=0.8, 
                                  edgecolor='lightgray', 
                                  linewidth=1.0)
                ax.add_patch(circle)
                
                # Add node ID label below the circle
                ax.text(x, y - radius - max_radius * 0.3, str(node_id), ha='center', va='top', 
                       fontsize=8, fontweight='bold')

        # Plot inactive nodes as small gray circles
        for node_id in range(N):
            if node_id not in node_error_dict or node_error_dict[node_id] == 0:
                x, y = adjusted_pos[node_id]
                radius = np.sqrt(min_circle_size / np.pi)
                circle = plt.Circle((x, y), radius, 
                                  color='lightgray', 
                                  alpha=0.3, 
                                  edgecolor='lightgray', 
                                  linewidth=1.0)
                ax.add_patch(circle)
                
                # Add node ID label below the circle
                ax.text(x, y - radius - max_radius * 0.3, str(node_id), ha='center', va='top', 
                       fontsize=8, fontweight='bold', color='gray')

        # Set equal aspect ratio and adjust limits
        ax.set_aspect('equal')
        
        # Calculate plot limits based on circle positions and sizes
        all_x = [adjusted_pos[i][0] for i in range(N)]
        all_y = [adjusted_pos[i][1] for i in range(N)]
        
        margin = max_radius * 2
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin - max_radius * 0.5, max(all_y) + margin)  # Extra space for labels

        # Add colorbar with proper sizing
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        
        # Position colorbar at bottom right
        ax_pos = ax.get_position()
        cbar_ax = fig.add_axes([ax_pos.x1 - 0.15, ax_pos.y0, 0.12, 0.03])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Inverted Normalised Error', fontsize=10)

        # Create legend for circle sizes
        legend_sizes = [min_count, (min_count + max_count) // 2, max_count]
        legend_circles = []
        legend_labels = []
        
        for count in legend_sizes:
            if max_count > min_count:
                normalized_count = (count - min_count) / (max_count - min_count)
            else:
                normalized_count = 0
            size = min_circle_size + normalized_count * (max_circle_size - min_circle_size)
            radius = np.sqrt(size / np.pi)
            legend_circles.append(plt.Circle((0, 0), radius, color='lightblue', alpha=0.7))
            legend_labels.append(f'{count} visits')

        # Position legend properly within the plot area
        # Position legend at bottom left
        legend_ax = fig.add_axes([ax_pos.x0, ax_pos.y0, 0.15, ax_pos.height * 0.25])
        legend_ax.set_xlim(-50, 50)
        legend_ax.set_ylim(-100, 100)
        legend_ax.set_aspect('equal')
        legend_ax.axis('off')

        y_positions = [-60, 0, 60]
        for i, (circle, label) in enumerate(zip(legend_circles, legend_labels)):
            circle.center = (0, y_positions[i])
            legend_ax.add_patch(circle)
            legend_ax.text(25, y_positions[i], label, ha='left', va='center', fontsize=8)

        legend_ax.text(0, 80, 'Node Visits', ha='center', va='center', fontsize=10, fontweight='bold')

        # Set main plot properties
        ax.set_xlabel('X (A.U.)', fontsize=12)
        ax.set_ylabel('Y (A.U.)', fontsize=12)
        ax.grid(True, alpha=0.3)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="both", which="minor", labelsize=6)

        plt.tight_layout()
        plt.savefig("node_map_topdown.pdf")
        plt.show()


    def plot_convergence(self, data):
        cycles_means = []
        cycles_stds = []
        max_steps = max(len(cycle) for cycle in data)

        for cycle in data:
            means = []
            stds = []
            for step in cycle:
                arr = np.array(step)
                means.append(np.mean(arr))
                stds.append(np.std(arr))
            cycles_means.append(means)
            cycles_stds.append(stds)

        # Pad sequences with NaN for missing steps
        cycles_means_padded = [np.pad(m, (0, max_steps - len(m)), constant_values=np.nan) for m in cycles_means]
        cycles_stds_padded = [np.pad(s, (0, max_steps - len(s)), constant_values=np.nan) for s in cycles_stds]

        # Convert to arrays for easier aggregation
        cycles_means_arr = np.array(cycles_means_padded)
        cycles_stds_arr = np.array(cycles_stds_padded)

        # Calculate overall central tendency and variability across cycles
        mean_over_cycles = np.nanmean(cycles_means_arr, axis=0)
        std_over_cycles = np.nanstd(cycles_means_arr, axis=0)

        steps = np.arange(0, max_steps)

        fig, ax = plt.subplots(1,1, layout="constrained")
        ax.plot(steps, mean_over_cycles, label='Mean error across repetitions', color='blue')
        ax.fill_between(steps,
                        mean_over_cycles - std_over_cycles,
                        mean_over_cycles + std_over_cycles,
                        color='blue', alpha=0.2)

        self._configure_axis(ax,"Error","Steps")
        self._set_small_fig(fig,ax)

        plt.savefig("convergence.pdf")
        plt.show()