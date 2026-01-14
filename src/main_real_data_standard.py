import numpy as np
from metricTensorOptimisation.tensor_optimisation_standard import tensorOptimisation

from metricTensorOptimisation.groupLevelAnalysis import GroupLevelAnalysis

from metricTensorOptimisation.plotting_functions_v2 import PlottingFunctions

if __name__ == '__main__':

    np.random.seed(42)

    ROI = [44, 48, 49, 58, 59] # non-pythonic indexing

    param_args = np.array([[0.05,0.001], # lower bounds
                           [2.0,0.05], # upper bounds
                           [.5,.5], # perturbation
                           [.1,.1] # learning rate
                           ])

    runType = "group"
    runSimulation = False
    groupSize = 20

    if runType == "single":

        optimiser = tensorOptimisation(param_args, f'data\\real_data\\*', ROI, num_particles=10, name_prefix="real_single")
    
        if runSimulation == True:
            optimiser.run(num_cycles=1, print_results=True, max_steps=15, auto_save=True)
    
        analyser = GroupLevelAnalysis()
        analyser.print_particle_history("output\\optimisation_results\\real_single_funcs_7_cycle", 1)

        plotter = PlottingFunctions(optimiser)
        plotter.plot_optimisation_results("output\\optimisation_results\\real_single_funcs_7_cycle1.pkl", channels=ROI)
        # plotter.plot_optimisation_results_mean_channel("output\\optimisation_results\\real_single_funcs_7_cycle1.pkl", channels=ROI)
        # plotter.plot_param_history("output\\optimisation_results\\real_single_funcs_7_cycle1.pkl")

    elif runType == "group":
        
        optimiser = tensorOptimisation(param_args, f'data\\real_data\\*', ROI, num_particles=20, name_prefix="real_group_standard")
    
        if runSimulation == True:
            optimiser.run(num_cycles=20, print_results=True, max_steps=15, auto_save=True, leave_one_out=True, error_threshold=4)

        analyser = GroupLevelAnalysis()

        analyser.unprocessed_check(optimiser, a=10, b=0, c=0, d=10)

        analyser.print_convergence_results("output\\optimisation_results\\real_group_standard_funcs_7_loo_index", optimiser)

        # analyser.compare_pipelines("output\\optimisation_results\\real_group_standard_funcs_7_loo_index", optimiser)

        history, error = analyser.return_particle_history("output\\optimisation_results\\real_group_standard_funcs_7_loo_index")
        plotter = PlottingFunctions(optimiser)

        # plotter.plot_optimisation_results("output\\optimisation_results\\real_group_standard_funcs_7_loo_index5.pkl", channels=ROI)

        print(error)

        plotter.plot_convergence(error)

        plotter.plot_node_map(history, error, "output\\adjacency_matrices\\adj_matrix_7.pkl")
