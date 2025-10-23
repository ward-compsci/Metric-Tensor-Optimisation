import numpy as np
import networkx as nx
import glob
import os
import pickle
import time

from snirf import Snirf

from metricTensorOptimisation.processing_functions_v2 import signalProcessing
from metricTensorOptimisation.graph_formation import graphWrapping
from metricTensorOptimisation.particle_structures import *

class tensorOptimisation:
    def __init__(self, param_args, files_location, ROI, num_particles=1, save_dir="output\\", name_prefix=None):
        """
        For n parameters, pass in a params array of form
        param_args = np.array([
            [lower_bound_1, ..., lower_bound_n],
            [upper_bound_1, ..., upper_bound_n],
            [perturbation_1, ..., perturbation_n],
            [learning_rate_1, ..., learning_rate_n]
            ])
        This assumes the order of functions -> as listed in signalProcessing
            matches the order of the parameters input
        """

        self.param_bounds = param_args[0:2,:]
        self.param_delta = param_args[2,:]
        self.learning_rate = param_args[3,:]

        self.num_particles = num_particles

        self.ROI_channels = ROI

        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # ================ #
        self.snirf_files = []
        for file in glob.glob(files_location):
            self.snirf_files.append(Snirf(file, 'r'))
            print(file)

        self.num_participants = len(self.snirf_files)

        self.extract_snirf_metaData(self.snirf_files[0])
        self.extract_snirf_measurementData(); print(f'=== Cohort datasets loaded ===')
      
        self.processing_instance = signalProcessing(self)
        self.graph_instance = graphWrapping(self.processing_instance, save_dir=save_dir) # wraps the functions to identifiers and intialises the graph
        
        # ================ #

        self.name_prefix = f"{name_prefix}_" if name_prefix else ""

        file_path = os.path.join(self.save_dir, f"graph\\graph_{self.graph_instance.num_functions}.pkl")
        
        if os.path.exists(file_path):
            print(f"Loading pre-saved graph for {self.graph_instance.num_functions} functions...")
            with open(file_path, "rb") as f:
                self.graph = pickle.load(f)
            print(f"Loaded pre-saved graph for {self.graph_instance.num_functions} functions")
        else:
            print(f"Generating graph for {self.graph_instance.num_functions} functions")
            self.graph = nx.from_scipy_sparse_array(self.graph_instance.adjacency_matrix)
            print(f"{self.graph} generated for {self.graph_instance.num_functions} functions")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump((self.graph), f)
        
        # ================ #
        self.param_count = list(self.graph_instance.arg_count_map.values())
        self.num_params = sum(self.param_count)


    def extract_snirf_metaData(self, snirf_file):

        """ ============== src det locations ============== """

        src_pos = snirf_file.nirs[0].probe.sourcePos3D
        det_pos = snirf_file.nirs[0].probe.detectorPos3D

        num_measurements = len(snirf_file.nirs[0].data[0].measurementList)

        src_indices = np.zeros([num_measurements]).astype(int); det_indices = np.zeros([num_measurements]).astype(int)

        for i,ml in enumerate(snirf_file.nirs[0].data[0].measurementList):
            src_indices[i] = ml.sourceIndex; det_indices[i] = ml.detectorIndex

        """ ============== SD dictionaries ============== """
        self.channel_dict = {} # we store SD pairs, and their distances in dictionaries where the key is the channel number
        self.channel_distance_dict = {}
        self.channel_location_dict = {}  # Store channel midpoints
        
        self.short_channels =  []

        for i in range(0, num_measurements, 2):
            src = src_pos[src_indices[i]-1,:]
            det = det_pos[det_indices[i]-1,:]

            channel_num = int(i / 2) + 1
            self.channel_dict[channel_num] = (src_indices[i], det_indices[i])

            distance = np.linalg.norm(src - det)
            midpoint = (src + det) / 2  # Compute the channel midpoint

            self.channel_distance_dict[channel_num] = distance
            self.channel_location_dict[channel_num] = midpoint

            if distance < 0.01:
                self.short_channels.append(channel_num)

        self.num_channels = int(np.shape(snirf_file.nirs[0].data[0].dataTimeSeries)[1] / 2)
        self.num_ROI_channels = len(self.ROI_channels)
        self.ROI_channels_index = np.array(self.ROI_channels)-1 # convert to pythonic indexing
   
        self.wavelength_labels = snirf_file.nirs[0].probe.wavelengths

    def extract_snirf_measurementData(self):
        """ ==================== Re-formatting data ==================== """
        """
        We have the following general variables:
            - time
            - num_timepoints
            - fs
            - num_channels
            - num_ROI_channels
        """

        self.participants_time_series = []

        max_timepoints = 0
        min_timepoints = len(self.snirf_files[0].nirs[0].data[0].time)
        max_time_snirf = self.snirf_files[0]
        min_time_snirf = self.snirf_files[0]

        # Find max timepoints across all participants
        for snirf in self.snirf_files:
            self.participants_time_series.append(snirf.nirs[0].data[0].time)
            num_timepoints = snirf.nirs[0].data[0].dataTimeSeries.shape[0]
            if num_timepoints > max_timepoints:
                max_timepoints = num_timepoints
                max_time_snirf = snirf  # Store the SNIRF file with max timepoints
            if num_timepoints < min_timepoints:
                min_timepoints = num_timepoints
                min_time_snirf = snirf

        self.num_timepoints = min_timepoints
        self.time = min_time_snirf.nirs[0].data[0].time  # Assign corresponding time array

        self.fs = 1 / (self.time[1] - self.time[0])

        self.num_timepoints = min_timepoints  # Update num_timepoints
    
        # Create combined channel list : ROI + short channels
        self.combined_channels = self.ROI_channels + self.short_channels
        self.num_combined_channels = len(self.combined_channels)

        # Create mapping from combined index to original channel
        self.combined_to_original = {i: ch for i, ch in enumerate(self.combined_channels)}

        # Update ROI channel indices for combined tensor
        self.ROI_channels_index_combined = np.arange(len(self.ROI_channels))
        self.short_channels_index_combined = np.arange(len(self.short_channels)) + len(self.ROI_channels)

        # Allocate array for combined channels only
        self.snirf_data = np.zeros([self.num_timepoints, self.num_combined_channels, 2, self.num_participants])

        # Update short channel mapping for combined tensor
        self.short_channel_indexes = np.zeros([len(self.ROI_channels), self.num_participants]).astype(int)

        for p in range(self.num_participants):
            data = self.snirf_files[p].nirs[0].data[0].dataTimeSeries
            time_len = data.shape[0] # Actual length of participant's data

            # Extract data for combined channels
            for combined_idx, original_ch in enumerate(self.combined_channels):
                # Get wavelength 1 and 2 indices for this channel in the original data
                wv1_idx = (original_ch - 1) * 2
                wv2_idx = wv1_idx + 1

                self.snirf_data[:, combined_idx, 0, p] = data[:self.num_timepoints, wv1_idx]
                self.snirf_data[:, combined_idx, 1, p] = data[:self.num_timepoints, wv2_idx]

            # Update short channel mapping for ROI channels in combined tensor
            if self.short_channels:
                for roi_idx, roi_ch in enumerate(self.ROI_channels):
                    roi_location = self.channel_location_dict[roi_ch]
                    closest_short_channel = min(self.short_channels, key=lambda sc: np.linalg.norm(roi_location - self.channel_location_dict[sc]))

                # Find the index of the closest short channel in the combined tensor
                short_ch_combined_idx = self.combined_channels.index(closest_short_channel)
                self.short_channel_indexes[roi_idx, p] = short_ch_combined_idx

        print(f"ROI channels: {self.ROI_channels}")
        print(f"Short channels: {self.short_channels}")

        print(f"Combined channels: {self.combined_channels}")
        print(f"Num combined channels: {self.num_combined_channels}")
        print(f"Map: {self.combined_to_original}")
        print(f"ROI channels index combined: {self.ROI_channels_index_combined}")
        print(f"Short channels index combined: {self.short_channels_index_combined}")
        print(f"Data shape: {self.snirf_data.shape}")

        print(f"Short channel indexes: {self.short_channel_indexes}")

        print(f"{[ch for ch in np.arange(1, len(self.combined_channels)+1)]}")
        print(f"{[self.channel_distance_dict[ch] * 1e2 * 1 for ch in np.arange(1,len(self.combined_channels)+1)]}")

        # # Allocate zero-padded array
        # self.snirf_data = np.zeros([self.num_timepoints, self.num_channels, 2, self.num_participants])

        # self.short_channel_indexes = np.zeros([self.num_channels, self.num_participants]).astype(int)
        """
        for p in range(self.num_participants):
            data = self.snirf_files[p].nirs[0].data[0].dataTimeSeries
            time_len = data.shape[0]  # Actual length of this participant's data

            wv1_counter = 0
            wv2_counter = 0

            for i in range(self.num_channels * 2):  # Reformat to [time x channel x wavelength]
                if self.snirf_files[p].nirs[0].data[0].measurementList[i].wavelengthIndex == 1:
                    self.snirf_data[:, wv1_counter, 0, p] = data[:self.num_timepoints, i]  # Copy valid data
                    wv1_counter += 1
                elif self.snirf_files[p].nirs[0].data[0].measurementList[i].wavelengthIndex == 2:
                    self.snirf_data[:, wv2_counter, 1, p] = data[:self.num_timepoints, i]  # Copy valid data
                    wv2_counter += 1

            if self.short_channels:
                for ch in range(self.num_channels):  # Iterate over long channels
                    ch_location = self.channel_location_dict[ch+1]
                    closest_short_channel = min(self.short_channels, key=lambda sc: np.linalg.norm(ch_location - self.channel_location_dict[sc]))

                    self.short_channel_indexes[ch,p] = int(closest_short_channel - 1)
        """

        buffer_seconds = 20
        end_time = self.time[-1]

        self.stim_start = []
        for snirf_file in self.snirf_files:
            stim_times = snirf_file.nirs[0].stim[0].data[:, 0]
            # Filter stim events that start before end_time - buffer_seconds
            stim_times_filtered = stim_times[stim_times <= (end_time - buffer_seconds)]
            self.stim_start.append(stim_times_filtered)
            # self.stim_start.append(snirf_file.nirs[0].stim[0].data[:,0])

        min_num_stim_events = min(len(stim) for stim in self.stim_start)
        
        truncated_stim_data = np.array([stim[:min_num_stim_events] for stim in self.stim_start])

        self.stim_data = truncated_stim_data
        self.num_stim_events = np.shape(self.stim_data)[1]

    def close_snirf_files(self):
        """Ensure all Snirf files are closed before multiprocessing."""
        if hasattr(self, "snirf_files"):
            for snirf in self.snirf_files:
                if hasattr(snirf, "file"):
                    snirf.file.close()  # Close HDF5 file handle
            self.snirf_files = None  # Remove reference

    def vectors_random(self):
        random_move_vector = np.zeros_like(self.particles)
        for particle in range(self.num_particles):
            current_node = int(self.particles[particle,0])
            neighbours = list(self.graph.neighbors(current_node))
            random_move_vector[particle,0] = np.random.choice(neighbours)
            
            for i in range(self.num_params):    
                random_direction = np.random.choice([-1,1])
                random_move_vector[particle,i+1] = self.particles[particle,i+1] + (random_direction * self.param_delta[i])
        
        return random_move_vector

    def vectors_local(self):
        local_vector = np.zeros_like(self.particles)
        for i, particle in enumerate(self.particles):
            # delta_step_nodes, gradient_nodes = self.finite_difference_nodes(particle)
            # gradient_params = self.finite_difference_params(particle)
            delta_step_nodes, gradient_nodes, gradient_params = self.finite_difference(particle)

            best_target_node = self.move_node(gradient_nodes, delta_step_nodes, particle)
            updated_params = particle[1:] - self.learning_rate * gradient_params

            local_vector[i,0] = best_target_node
            local_vector[i,1:] = updated_params

        return local_vector

    def vectors_global(self):
        particle_score = np.zeros([self.num_particles])

        particle_score = self.current_error

        best_particle_idx = np.argmin(particle_score)
        if min(particle_score) == self.num_channels:
            best_particle_idx == np.random.randint(0,self.num_particles+1).astype(int) # if none are correct, choose a random one

        best_particle = self.particles[best_particle_idx,:]

        global_vector = np.zeros_like(self.particles)

        for i, particle in enumerate(self.particles):
            all_paths = list(nx.all_shortest_paths(self.graph, source=particle[0], target=best_particle[0]))
            if len(all_paths) == 1:
                global_vector[i,0] = best_particle[0]
            else:
                second_elements = [path[1] for path in all_paths]
                selected_path = np.random.choice(second_elements)
                global_vector[i,0] = selected_path

            distance = particle[1:] - self.particles[best_particle_idx, 1:]
            global_vector[i, 1:] = particle[1:] - self.learning_rate * distance

        return global_vector

    def total_vector(self):
        random_vector, local_vector, global_vector = self.vectors_random(), self.vectors_local(), self.vectors_global()

        stacked_vectors = np.stack([random_vector, local_vector, global_vector], axis=1)

        particles_vector = np.zeros_like(self.particles)

        particles_vector[:,0] = np.apply_along_axis(self.custom_mode, 1, stacked_vectors[:,:,0])
        particles_vector[:,1:] = np.mean(stacked_vectors[:,:,1:], axis=1)
        particles_vector[:,1:] = np.clip(particles_vector[:,1:], self.param_bounds[0,:], self.param_bounds[1,:])

        swarm_proposed_particles = ParticleSwarm(self.graph_instance)
        swarm_proposed_particles.create_particles_from_array(particles_vector)
        proposed_signals = swarm_proposed_particles.run_parallel(self.snirf_data)

        # Enforce 'no worse state' rule
        for i, particle in enumerate(self.particles):
            proposed_error = self.is_active(proposed_signals[i])

            if proposed_error > self.current_error[i]:
                particles_vector[i] = particle # Keep particle in current state

        return particles_vector


    def custom_mode(self, values):
        unique_vals,counts = np.unique(values, return_counts=True)
        max_count = np.max(counts)
        candidates = unique_vals[counts == max_count]
        return np.random.choice(candidates)

    def get_n_step_nodes(self, node, n):
        lengths = nx.single_source_shortest_path_length(self.graph, node)
        return [node for node, dist in lengths.items() if dist == n]
    
    def find_shortest_paths(self, start_node, target_node):
        
        return list(nx.shortest_path(self.graph, source=start_node, target=target_node))

    def move_node(self, gradients, nodes, particle):
        
        indexes = np.where(gradients == gradients.min())
        best_nodes = [nodes[i] for i in indexes]
        if min(gradients[indexes]) >= 0:
            best_node = particle[0]
        elif np.shape(indexes)[1] > 1:
            best_node = np.random.choice(best_nodes[0])
        else:
            best_node = best_nodes[0]

        return best_node
    
    def finite_difference(self, particle, delta=1):
        dC_dNode = []
        dC_dParam = []

        # Current node and params
        current_node = int(particle[0])
        current_params = particle[1:]

        # Get n-step nodes and initialise particle
        n_step_nodes = self.get_n_step_nodes(current_node, delta)
        n_step_nodes.append(current_node)

        current_particle = Particle(current_node, current_params, self.graph_instance)
        current_error = self.is_active(current_particle.run(self.snirf_data))

        if not n_step_nodes: # no nodes are delta steps away
            return []
        
        # Prepare particles for finite difference of nodes
        particle_list_nodes = []
        for n_step_node in n_step_nodes:
            particle_list_nodes.append([n_step_node] + current_params.tolist())

        # Prepare particles for finite difference of params
        positive_particles = []
        negative_particles = []
        for i in range(len(current_params)):
            param_plus, param_minus = current_params.copy(), current_params.copy()
            param_plus[i] = np.clip(param_plus[i] + self.param_delta[i], self.param_bounds[0,i], self.param_bounds[1,i])
            param_minus[i] = np.clip(param_plus[i] - self.param_delta[i], self.param_bounds[0,i], self.param_bounds[1,i])

            positive_particles.append([current_node] + param_plus.tolist())
            negative_particles.append([current_node] + param_minus.tolist())

        # Combine all particles (node and param)
        total_particles = particle_list_nodes + positive_particles + negative_particles
        swarm = ParticleSwarm(self.graph_instance)
        swarm.create_particles_from_list(total_particles)
        signals = swarm.run_parallel(self.snirf_data)

        # Separate signals for nodes and parameters
        new_signals_nodes = signals[:len(particle_list_nodes)]
        positive_signals = signals[len(particle_list_nodes):len(particle_list_nodes) + len(positive_particles)]
        negative_signals = signals[len(particle_list_nodes) + len(positive_particles):]

        # Process node finite difference results
        for solution in range(len(new_signals_nodes)):
            neighbour_error = self.is_active(new_signals_nodes[solution])
            dB_dNode = (neighbour_error - current_error) / delta
            dC_dNode.append(current_error * dB_dNode)
            
        # Process parameter finite difference results
        for i in range(len(current_params)):
            B_plus_error = self.is_active(positive_signals[i])
            B_minus_error = self.is_active(negative_signals[i])

            dB_dParam = (B_plus_error - B_minus_error) / (2 * self.param_delta[i])
            dC_dParam.append(current_error * dB_dParam)

        return np.asarray(n_step_nodes), np.asarray(dC_dNode), np.asarray(dC_dParam)


    def is_active(self, signals, threshold=50, stats_duration=0.5, expected_peak=5.9, activation_method="group", haemodynamics_check="single", print_result=False):
        """
        Check activation of ROI for a cohort given two possible methods:
        1. "group" (default) : Block average statistics per channel.
        2. "per_block" : Performs t-test per stimulus block and determines activation given number of active blocks per channel
        3. "single_block" : Performs t-test just over the first block

        TODO: check if t-stat contains infs/nan and deal with them
        """

        stats_post_start = expected_peak - (stats_duration / 2)
        stats_post_start_2 = expected_peak * 2 - (stats_duration / 2)


        def find_nearest(a, a0):
            "Element in nd array `a` closest to the scalar value `a0`"
            idx = np.abs(a - a0).argmin()
            return idx
 
        error = 0

        # Use ROI indices in combined tensor
        signals_for_testing = signals[:,self.ROI_channels_index_combined,:,:]

        t_statistic_cohort = np.zeros([self.num_ROI_channels,2,self.num_participants, 2])

        stim_time = np.linspace(-stats_duration, stats_post_start_2 + stats_duration,
                                int((stats_post_start_2 + stats_duration*1.5) * self.fs))
        
        stim_time_num = len(stim_time)
        t0_index = find_nearest(stim_time, 0)

        for participant in range(self.num_participants):
            stim_blocks = np.zeros([stim_time_num, self.num_ROI_channels, 2, self.num_stim_events])

            for i, stimulus in enumerate(self.stim_data[participant,:]):
                stim_idx = find_nearest(self.time, stimulus - stats_duration)
                stim_block = signals_for_testing[np.ix_(
                    np.arange(stim_idx, stim_idx+stim_time_num),
                    np.arange(self.num_ROI_channels),
                    np.arange(2),
                    [participant]
                )] # numpy advanced vs simple indexing - it swaps dimensions
                stim_block = stim_block.squeeze(-1)
                stim_blocks[:,:,:,i] = stim_block - stim_block[t0_index,:,:]

            # Pre-stimulus period
            pre_mask = (stim_time >= - stats_duration) & (stim_time < 0)
            num_samples = np.sum(pre_mask)

            # Maximum response period
            mid_start_index = find_nearest(stim_time, stats_post_start); final_start_index = find_nearest(stim_time, stats_post_start_2)
            
            mid_mask = np.zeros_like(stim_time, dtype=bool); final_mask = np.zeros_like(stim_time, dtype=bool)
            for i in range(num_samples):
                mid_mask[mid_start_index+i] = True
                final_mask[final_start_index+i] = True
            
            if activation_method == "group":

                stim_blocks_mean = np.mean(stim_blocks,axis=-1)

                average_differences_pre_mid = np.mean(stim_blocks_mean[pre_mask,:,:] - stim_blocks_mean[mid_mask,:,:],axis=0)
                std_differences_pre_mid = np.std(stim_blocks_mean[pre_mask,:,:] - stim_blocks_mean[mid_mask,:,:],axis=0)
                t_statistic_pre_mid = average_differences_pre_mid / (std_differences_pre_mid / np.sqrt(num_samples))

                average_differences_mid_final = np.mean(stim_blocks_mean[mid_mask,:,:] - stim_blocks_mean[final_mask,:,:],axis=0)
                std_differences_mid_final = np.std(stim_blocks_mean[mid_mask,:,:] - stim_blocks_mean[final_mask,:,:],axis=0)
                t_statistic_mid_final = average_differences_mid_final / (std_differences_mid_final / np.sqrt(num_samples))

                for channel in range(self.num_ROI_channels):
                    if haemodynamics_check == "single":
                        if t_statistic_pre_mid[channel, 0] < -threshold and t_statistic_mid_final[channel, 0] > threshold:  # Channel is active
                            if print_result == True:
                                print(f"Participant {participant+1} channel {channel+1} is active")
                            break  # Exit the loop early if any channel is active
                    elif haemodynamics_check == "all":
                        if t_statistic_pre_mid[channel, 0] < -threshold and t_statistic_mid_final[channel, 0] > threshold:  # Channel is active
                            if t_statistic_pre_mid[channel,1] > threshold and t_statistic_mid_final[channel,1] < - threshold:
                                if print_result == True:
                                    print(f"Participant {participant+1} channel {channel+1} is active")
                                break  # Exit the loop early if any channel is active
                    else:
                        raise ValueError("haemodynamics_check must be 'single' or 'all'")
                else:
                    # If the loop completes without 'break', it means no channels are active
                    error += 1  # Add 1 error for no active channels

                t_statistic_cohort[:,:,participant, 0] = t_statistic_pre_mid
                t_statistic_cohort[:,:,participant, 1] = t_statistic_mid_final


            if activation_method not in ["group"]:
                raise ValueError(f"Invalid activation method: '{activation_method}'. Choose 'group'.")

            

        return error

    def run_cycle(self, max_steps=15, print_results=False, stopping_error_threshold=4, evaluation_data=None):
        """
        Runs a single optimisation cycle. Stores internal histories and returns all results and metadata
        """

        try:
            self.close_snirf_files()
        except:
            pass

        swarm = ParticleSwarmInitial(self.num_particles, self.graph_instance, self.param_bounds)
        self.particles = swarm.get_particles()

        self.error_history = []
        self.particles_history = []

        if print_results:
            print(f"Initial particles:\n{self.particles}")

        # Run initial signal reconstruction
        initial_signals = swarm.run_parallel(self.snirf_data)
        initial_errors = [self.is_active(initial_signals[i]) for i in range(self.num_particles)]

        self.particles_history.append(self.particles)
        self.error_history.append(initial_errors)

        self.current_error = initial_errors
        self.best_particle_error = [min(initial_errors)]
        
        # Initialise signals with initial signals
        signals = initial_signals

        step = 0
        found_solution = False

        # Early stopping condition on initial step
        if self.best_particle_error[-1] <= stopping_error_threshold:
            found_solution = True
            step = 0
            print(f"Found solution on initialisation")

        while not found_solution and step < max_steps:
            step += 1
            print(f"Step {step}")
            
            movement_vector = self.total_vector()
            self.particles = movement_vector

            swarm.create_particles_from_array(self.particles)
            signals = swarm.run_parallel(self.snirf_data)

            iteration_errors = [self.is_active(signals[i]) for i in range(self.num_particles)]

            self.error_history.append(iteration_errors)
            self.particles_history.append(self.particles)

            self.current_error = iteration_errors
            best_error = min(iteration_errors)
            self.best_particle_error.append(best_error)

            print(f"Best particle error: {best_error}")

            if best_error <= stopping_error_threshold:
                found_solution = True

        # Final evaluation on full cohort if in LOO
        if evaluation_data is not None:
            eval_data, eval_stim, eval_n = evaluation_data

            self.snirf_data = eval_data
            self.stim_data = eval_stim
            self.num_participants = eval_n

            signals = swarm.run_parallel(self.snirf_data)
            eval_errors = [self.is_active(signals[i]) for i in range(self.num_particles)]
        else:
            eval_errors = self.current_error

        # === Build metadata ===
        meta = {
            "run_name": self.name_prefix,
            "num_particles": self.num_particles,
            "num_functions": self.graph_instance.num_functions,
            "final_error": self.best_particle_error[-1],
            "eval_error": min(eval_errors),
            "steps": step,
            "timestamp": time.time()
        }

        results = {
            "Particle history": self.particles_history,
            "Error history": self.error_history,
            "Reconstructed signals": signals,
            "Meta": meta
        }

        return results

    def save_results(self, results_dict, suffix=None, filename=None):
        """
        Save a dictionary of results to disk, including metadata.
        :param results_dict: dictionary from run_cycle()
        :param suffix: optional string to append to filename
        :param filename: fully custom filename (overrides suffix)
        """

        folder_path = os.path.join(self.save_dir, "optimisation_results")
        os.makedirs(folder_path, exist_ok=True)

        if filename:
            file_path = os.path.join(folder_path, filename)
        else:
            suffix = f"_{suffix}" if suffix else ""
            file_path = os.path.join(
                folder_path,
                f"{self.name_prefix}funcs_{self.graph_instance.num_functions}{suffix}.pkl"
            )

        with open(file_path, "wb") as f:
            pickle.dump(results_dict, f)

        print(f"Saved results to {file_path}")

    def run(self, num_cycles=1, print_results=False, max_steps=15, error_threshold=4, auto_save=True, leave_one_out=False):
        """
        Run multiple optimisation cycles or leave-one-out evaluation.
        Returns a list of result dictionaries.
        """

        all_results = []
        folder_path = os.path.join(self.save_dir, "optimisation_results")
        os.makedirs(folder_path, exist_ok=True)

        if leave_one_out:
            print(f"\n=== Running Leave-One-Out over {self.num_participants} participants ===")
            for loo_idx in range(self.num_participants):
                suffix = f"loo_index{loo_idx+1}"
                expected_filename = f"{self.name_prefix}funcs_{self.graph_instance.num_functions}_{suffix}.pkl"
                expected_filepath = os.path.join(folder_path, expected_filename)

                if auto_save and os.path.exists(expected_filepath):
                    print(f"=== Skipping cycle {loo_idx+1}/{self.num_participants} (already exists) ===")
                    continue

                print(f"\n=== Starting Leave-One-Out {loo_idx+1}/{self.num_participants} (leaving out participant {loo_idx+1}) ===")

                # Backup full dataset
                full_data = self.snirf_data.copy()
                full_stim = self.stim_data.copy()
                full_n = self.num_participants

                # Train without one participant
                self.snirf_data = np.delete(self.snirf_data, loo_idx, axis=-1)
                self.stim_data = np.delete(self.stim_data, loo_idx, axis=0)
                self.num_participants = self.snirf_data.shape[-1]

                results = self.run_cycle(
                    max_steps=max_steps,
                    print_results=print_results,
                    stopping_error_threshold=error_threshold,
                    evaluation_data=(full_data, full_stim, full_n)
                )


                all_results.append(results)

                if auto_save:
                    self.save_results(results, suffix=suffix)

                # Restore full data
                self.snirf_data = full_data
                self.stim_data = full_stim
                self.num_participants = full_n

        else:


            for cycle in range(num_cycles):
                suffix = f"cycle{cycle+1}"
                expected_filename = f"{self.name_prefix}funcs_{self.graph_instance.num_functions}_{suffix}.pkl"
                expected_filepath = os.path.join(folder_path, expected_filename)

                if auto_save and os.path.exists(expected_filepath):
                    print(f"=== Skipping cycle {cycle+1}/{num_cycles} (already exists) ===")
                    continue

                print(f"\n=== Starting optimisation cycle {cycle+1}/{num_cycles} ===")

                results = self.run_cycle(
                    max_steps=max_steps,
                    print_results=print_results,
                    stopping_error_threshold=error_threshold
                )

                all_results.append(results)

                if auto_save:
                    suffix = f"cycle{cycle+1}"
                    self.save_results(results, suffix=suffix)

        return all_results
