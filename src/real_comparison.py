import numpy as np

# Custom pipeline for real data using the same ROI as main_real_data.py
from metricTensorOptimisation.tensor_optimisation_v5 import tensorOptimisation
from metricTensorOptimisation.plotting_functions_v2 import PlottingFunctions

if __name__ == '__main__':
    # Select ROI (replace with the same ROI index or mask as used in main_real_data.py)
    ROI = [44, 48, 49, 58, 59] # non-pythonic indexing
    param_args = np.array([[0.05,0.001], # lower bounds
                            [2.0,0.05], # upper bounds
                            [.5,.5], # perturbation
                            [.1,.1] # learning rate
                            ])


    optimiser = tensorOptimisation(param_args, f'data\\real_data\\*', ROI, num_particles=1, name_prefix="real_single")

    print(optimiser.graph_instance.num2func_dict)
    print(optimiser.graph_instance.num2node_dict)

    # optimiser.particles[0] = [158, 0.01, 0.5] # (1,5,4,3,2)

    optimiser.run(num_cycles=1, print_results=True, max_steps=0, auto_save=True)

    plotter = PlottingFunctions(optimiser)
    # plotter.plot_optimisation_results_mean_channel("output\\optimisation_results\\real_single_funcs_7_cycle1.pkl", channels=ROI, print_plots=True, ylims=(0, 1.1))
    plotter.plot_optimisation_results_mean_channel(f"output\\optimisation_results\\real_single_funcs_7_cycle1.pkl", channels=ROI, print_plots=True)
