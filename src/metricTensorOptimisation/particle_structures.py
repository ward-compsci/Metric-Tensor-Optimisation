import numpy as np
import multiprocessing


class Particle:
    def __init__(self, node_id, params, graph_instance):
        self.node_id = node_id
        self.params = params
        self.graph_instance = graph_instance # to access function mappings

    def apply_functions(self, signals):
        """ Applies functions given particle node state. """
        function_identifiers = self.graph_instance.num2node_dict[self.node_id]
        processed_signals = signals.copy()

        param_count = list(self.graph_instance.arg_count_map.values())

        if function_identifiers == (0,):
            return processed_signals
                
        try:
            for i, func_id in enumerate(function_identifiers):
                function = self.graph_instance.function_map[func_id]
                num_params = param_count[func_id - 1]
                pre_params = sum(param_count[:func_id - 1])
                func_params = self.params[pre_params : pre_params + num_params]

                processed_signals = function(processed_signals, *func_params)

            return processed_signals
        
        except Exception as e:
            print(f'Error occured while applying function {func_id}. Composition : {function_identifiers}')
            print(f"{e}")
            return processed_signals


    def run(self, signals):
        return self.apply_functions(signals)
    
    def __repr__(self):
        return f"Node {self.node_id} : Params {self.params}"


def run_particle(particle, signals):
    """Standalone function to execute a particle, avoiding pickle issues."""
    return particle.run(signals)


class ParticleSwarm:
    def __init__(self, graph_instance):
        self.graph_instance = graph_instance
        self.particles = []
        self.num_particles = None

    def create_particles_from_list(self, particle_list):
        self.particles = []
        for particle_data in particle_list:
            node_id = particle_data[0]
            params = particle_data[1:]
            self.particles.append(Particle(node_id, params, self.graph_instance))

        self.num_particles = len(self.particles)

    def create_particles_from_array(self, particle_array):
        self.particles = []
        for particle_data in particle_array:
            node_id = int(particle_data[0])
            params = particle_data[1:]
            self.particles.append(Particle(node_id, params, self.graph_instance))

        self.num_particles = len(self.particles)

    def run_parallel(self, signals):
        """ Run particles in parallel with multiprocessing. """
        # print(f"Running {len(self.particles)} particles in parallel.")
    
        with multiprocessing.Pool(processes=4) as pool:
            results = pool.starmap(run_particle, [(p, signals) for p in self.particles])
        pool.close()
        pool.join()

        return results
    
    def get_particles(self):
        return np.array([np.hstack([p.node_id, p.params]) for p in self.particles])

    def __repr__(self):
        return f"Particle Swarm for {self.num_particles} particles"

class ParticleSwarmInitial(ParticleSwarm):
    def __init__(self, num_particles, graph_instance, param_bounds):
        super().__init__(graph_instance)
        self.num_particles = num_particles
        self.initialise_particles(param_bounds)

    def initialise_particles(self, param_bounds):
        """ Initialise particles with random parameters. """
        node_ids = list(self.graph_instance.num2node_dict.keys())
        particles = []

        for _ in range(self.num_particles):
            node_id = np.random.choice(node_ids)
            params = np.random.uniform(param_bounds[0,:], param_bounds[1,:], param_bounds.shape[1])
            particles.append([node_id] + list(params))

        self.create_particles_from_list(particles)
    
class ParticleWithHistory(Particle):
    """Particle that tracks its personal best position and error."""
    def __init__(self, node_id, params, graph_instance):
        super().__init__(node_id, params, graph_instance)
        self.personal_best_node = node_id
        self.personal_best_params = params.copy()
        self.personal_best_error = np.inf
    
    def update_personal_best(self, current_error):
        """Update personal best if current position is better."""
        if current_error < self.personal_best_error:
            self.personal_best_error = current_error
            self.personal_best_node = self.node_id
            self.personal_best_params = self.params.copy()
            return True
        return False