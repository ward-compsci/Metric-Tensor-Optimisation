import numpy as np

from metricTensorOptimisation.processing_functions_v2 import signalProcessing
from metricTensorOptimisation.graph_formation import graphWrapping
from metricTensorOptimisation.tensor_optimisation_v5 import tensorOptimisation

# from metricTensorOptimisation.plotting_functions import plottingFunctions
from metricTensorOptimisation.plotting_functions_v2 import PlottingFunctions

from metricTensorOptimisation.snirf_generator import snirfGenerator

from metricTensorOptimisation.groupLevelAnalysis import GroupLevelAnalysis

import pickle

import os

from snirf import Snirf


if __name__ == '__main__':

    np.random.seed(42)

    directory_path = f"data\\synthetic_data\\"

    ROI = [1, 3, 5]

    param_args = np.array([[0.05, 0.001],
                        [2.0, 0.05],
                        [0.5, 0.5],
                        [0.1, 0.1]])

    runType = "group"
    runSimulation = True
    groupSize = 20


    def create_snirf_file():
        ## Create synthetic dataset if does not exist
        if not os.listdir(directory_path):
            print(f"Generating SNIRF file")
            generator = snirfGenerator(channel_values=[1,0,1,0,1], duration=240, stim_events=[(10,2),(32,2),(53,2),(75,2),(95,2),
                                                                                            (116,2),(136,2),(155,2),(178,2),(200,2)])
            generator.plot_channel_intensity_with_hrf()
        else:
            pass


    if runType == "single":
        pass
    #     create_snirf_file()        

    #     optimiser = tensorOptimisation(param_args=param_args, files_location=f"data\\synthetic_data\\*", ROI=ROI, num_particles=10, name_prefix="synthetic_single")
    #     if runSimulation == True:
    #         optimiser.run(num_cycles=1, print_results=True, max_steps=15, auto_save=True)


    #     plotter = PlottingFunctions(optimiser)

    #     plotter.plot_optimisation_results(filepath="output\\optimisation_results\\synthetic_single_funcs_6_cycle1.pkl")


    elif runType == "group":

        create_snirf_file()

        optimiser = tensorOptimisation(param_args=param_args, files_location=f"data\\synthetic_data\\*", ROI=ROI, num_particles=10, name_prefix="synthetic_group")

        print(optimiser.graph)

        if runSimulation == True:
            optimiser.run(num_cycles=groupSize, print_results=True, max_steps=15, auto_save=True, error_threshold=0)


        analyser = GroupLevelAnalysis()

        analyser.unprocessed_check(optimiser, a=19, b=2, c=0, d=20)

        analyser.print_convergence_results("output\\optimisation_results\\synthetic_group_funcs_6_cycle", optimiser)
        # analyser.compare_pipelines("output\\optimisation_results\\synthetic_group_funcs_6_cycle", optimiser)

        history, error = analyser.return_particle_history("output\\optimisation_results\\synthetic_group_funcs_6_cycle")

        # print(error)

        plotter = PlottingFunctions(optimiser)

        # plotter.plot_convergence(error)

        # plotter.plot_optimisation_results(filepath="output\\optimisation_results\\synthetic_group_funcs_6_cycle6.pkl")
        # for i in range(1,11):
        #     plotter.plot_optimisation_results_mean_channel_synthetic(filepath=f"output\\optimisation_results\\synthetic_group_funcs_6_cycle{i}.pkl")

        plotter.plot_optimisation_results_mean_channel_synthetic(filepath="output\\optimisation_results\\synthetic_group_funcs_6_cycle3.pkl")

        plotter.plot_node_map(history, error, "output\\adjacency_matrices\\adj_matrix_6.pkl")
        # plotter.plot_node_map_topdown(history, error, "output\\adjacency_matrices\\adj_matrix_6.pkl")

        