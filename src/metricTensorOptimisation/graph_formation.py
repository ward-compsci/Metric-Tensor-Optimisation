import numpy as np
import itertools
import inspect
import re
import scipy.sparse as sp
from tqdm import tqdm  # Import tqdm for progress bars
import pickle
import os


class graphWrapping:
    def __init__(self, signal_processor_instance, save_dir):
        
        self.save_dir = save_dir  # Directory to save/load adjacency matrix
        # Create the directory if it doesn't exist
        
        self.adjacency_matrix = None
        self.node2num_dict = None
        self.num2node_dict = None

        self.assign_numeric_identifiers_functions(signal_processor_instance)
        self.generate_adjacency_matrix()
        

    def can_transition(self, node1, node2, min_node_length=1):
        """
        Determines if a transition between node1 and node2 is valid based on certain criteria.
        
        Parameters:
        - node1: list or array-like, representing the first node.
        - node2: list or array-like, representing the second node.
        
        Returns:
        - valid: Boolean indicating if the transition is valid (True) or not (False).
        """

        # Ensure (0,) connects to the shortest available nodes
        if node1 == (0,):
            return len(node2) == min_node_length  # Allow transition to the smallest valid length nodes
    
        
        # Check if one of the nodes is empty (sum of elements == 0)
        if ((sum(node1) == 0) ^ (sum(node2) == 0)):  # XOR condition in Python
            return len(node1) == len(node2) == 1
                
        # Lengths must differ by exactly 1
        if abs(len(node1) - len(node2)) != 1:
            return False
        
        # Identify shorter and longer
        shorter, longer = (node1, node2) if len(node1) < len(node2) else (node2, node1)

        # Check if loinger is exactly shorter with one element inserted
        for i in range(len(longer)):
            if longer[:i] + longer[i+1:] == shorter:
                return True
            
        return False


    def generate_adjacency_matrix(self):
        """
        Generates an adjacency matrix for a given number of functions based on transitions between nodes.

        Parameters:
        - num_functions: Integer, number of functions.

        Returns:
        - adjacency_matrix: 2D numpy array representing adjacency matrix.
        - node2num_dict: Dictionary mapping nodes to their numeric identifiers.
        - num2node_dict: Dictionary mapping numeric identifiers to nodes.
        """
        
        """ Efficiently generates adjacency matrix using only half computations. """

        # Check if the matrix is pre-generated
        file_path = os.path.join(self.save_dir, f"adjacency_matrices\\adj_matrix_{self.num_functions}.pkl")
        if os.path.exists(file_path):
            print(f"Loading pre-saved adjacency matrix for {self.num_functions} functions...")
            with open(file_path, "rb") as f:
                self.adjacency_matrix, self.node2num_dict, self.num2node_dict = pickle.load(f)
            print(f"Loaded pre-saved adjacency matrix for {self.num_functions} functions")
            return self.adjacency_matrix, self.node2num_dict, self.num2node_dict


        # Step 1: Generate all unique permutations, but only keep nodes that include both functions 1 and 2, and that have 1 before 2
        nodes = []
        for k in range(1, self.num_functions):
            for perm in itertools.permutations(range(1, self.num_functions), k):
                # Check if node contains both function 1 and function 2
                if 1 in perm and 2 in perm:
                    # Check if function 2 appears before function 1
                    if perm.index(2) > perm.index(1):
                        nodes.append(perm)

        nodes.insert(0, (0,))  # Add the empty node
        self.node2num_dict = {tuple(node): idx for idx, node in enumerate(nodes)}
        num_nodes = len(nodes)

        # Find the minimum length of non-empty nodes
        min_node_length = min(len(node) for node in nodes if node != (0,))

        # Step 2: Compute only the upper triangle (i < j)
        edges = []
        for i, node1 in tqdm(enumerate(nodes), total=num_nodes, desc="Computing nodes"):
            for j in range(i + 1, num_nodes):  # Only iterate over upper triangle
                node2 = nodes[j]
                if self.can_transition(node1, node2, min_node_length):
                    edges.append((i, j))  # Upper half only

        # Step 3: Build sparse adjacency matrix (full symmetry)
        row, col = zip(*edges) if edges else ([], [])
        adjacency_matrix = sp.csr_matrix((np.ones(len(row), dtype=int), (row, col)), shape=(num_nodes, num_nodes))

        # Ensure symmetry by adding the transpose
        self.adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        self.num2node_dict = {v: k for k, v in self.node2num_dict.items()}

        with open(file_path, "wb") as f:
            pickle.dump((self.adjacency_matrix, self.node2num_dict, self.num2node_dict), f)

        print(f"Adjacency matrix for {self.num_functions} functions saved.")

    
    def assign_numeric_identifiers_functions(self, instance):
        """
        Assigns numeric identifiers to instance methods in the order they are defined in the source code.

        Parameters:
        - instance: The class instance from which to retrieve methods.

        Returns:
        - function_map: A dictionary mapping numeric identifiers to the instance methods.
        - num2func_dict: A reverse dictionary mapping numeric identifiers to function names.
        """
        # Retrieve the source code of the class
        source = inspect.getsource(instance.__class__)

        # Use a regex to extract method names in the order of their definition
        method_names = re.findall(r'def\s+(\w+)\s*\(', source)

        # Initialize the function map
        function_map = {}
        num2func_dict = {}
        arg_count_map = {}

        # Start function numbering at 1
        func_id = 1
        for name in method_names:
            if name == "__init__":
                continue  # Skip the __init__ method
            
            # Get the actual method object from the instance
            method = getattr(instance, name, None)
            if method and inspect.ismethod(method):
                function_map[func_id] = method  # Assign the numeric identifier to the method
                num2func_dict[func_id] = name  # Store the method name for easier reference

                # Count the number of non-default arguments, excluding "signal"
                signature = inspect.signature(method)
                parameters = signature.parameters
                non_default_args = [
                    param for param in parameters.values()
                    if param.default == inspect.Parameter.empty and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                ]
                arg_count_map[func_id] = len(non_default_args) - 1  # Remove "signal" as a parameter
                
                func_id += 1  # Increment function id manually

        self.function_map = function_map
        self.num2func_dict = num2func_dict
        self.arg_count_map = arg_count_map
        self.num_functions = func_id

        return function_map, num2func_dict