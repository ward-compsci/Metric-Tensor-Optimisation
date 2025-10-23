import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.stats import fisher_exact


import networkx as nx
from collections import defaultdict
from itertools import combinations


from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch

import csv

import krippendorff
import pandas as pd

# TODO:
# Make it so the _load_cohort() checks for files beginning with the filename, and just iterates through them

class GroupLevelAnalysis:
    def __init__(self):
        pass

    def _load_cohort(self, filepath, num_files=10):
        files = []
        try:
            # while True:
            for i in range(num_files):
                with open(f"{filepath}{i+1}.pkl", "rb") as f:
                    files.append(pickle.load(f))
        except:
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return files
        
    def _load_cycle_file(self, filepath_prefix, cycle):
        filename = f"{filepath_prefix}{cycle}.pkl"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No file found at {filename}")
        with open(filename, "rb") as f:
            return pickle.load(f)

    def print_particle_history(self, filepath_prefix, cycle):
        file = self._load_cycle_file(filepath_prefix, cycle)

        particle_history = file["Particle history"]
        
        for step, particles in enumerate(particle_history):
            print(f"Step {step}:")
            print(np.array(particles))
            print()

    def return_particle_history(self, filepath):
        files = self._load_cohort(filepath)
        
        history = []
        error = []
        for file in files:
            particle_history = file["Particle history"]
            error_history = file["Error history"]
            temp = []
            for step, particles in enumerate(particle_history):
                temp.append(particles)
            history.append(temp)
            temp = []
            for step, errors in enumerate(error_history):
                temp.append(errors)
            error.append(temp)

        return history, error

    def print_convergence_results(self, filepath, parent):
        files = self._load_cohort(filepath)

        steps = []
        final_error = []
        eval_error = []
        particles = []
        pipelines = []

        for file in files:

            steps.append(file["Meta"]["steps"])
            final_error.append(file["Meta"]["final_error"])
            eval_error.append(file["Meta"]["eval_error"])
            best_particle = np.argmin(file["Error history"][-1])
            # best_particle = np.argmin(file["Meta"]["eval_error"][-1])
            particles.append(file["Particle history"][-1][best_particle])

        print(steps)
        print(final_error)
        print(eval_error)
        print(particles)

        print(f"Function mapping: \n{parent.graph_instance.num2func_dict}")
        for i in range(len(particles)):
            print(f"Best particle: {particles[i]}\nError: {final_error[i]}")
            print(f"Best particle combination: {parent.graph_instance.num2node_dict[int(particles[i][0])]}")


    def compare_pipelines(self, filepath, parent):
        files = self._load_cohort(filepath)

        particles = []
        pipelines = []

        for i, file in enumerate(files):
            best_particle = np.argmin(file["Error history"][-1])
            particles.append(file["Particle history"][-1][best_particle])
            pipelines.append(parent.graph_instance.num2node_dict[int(particles[i][0])])
        
        def lcs_length(a, b):
            """Compute the length of the longest common subsequence between sequences a and b."""
            dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
            for i in range(len(a)):
                for j in range(len(b)):
                    if a[i] == b[j]:
                        dp[i+1][j+1] = dp[i][j] + 1
                    else:
                        dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
            return dp[-1][-1]
        

        n = len(pipelines)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                lcs = lcs_length(pipelines[i], pipelines[j])
                # Normalize by average length (or max)
                norm = (len(pipelines[i]) + len(pipelines[j])) / 2
                sim_matrix[i, j] = lcs / norm


        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)

        # Axis labels
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([f"S{i}" for i in range(n)])
        ax.set_yticklabels([f"S{i}" for i in range(n)])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Title and labels
        ax.set_title("LCS-Based Similarity Between Pipelines")
        ax.set_xlabel("Pipeline Index")
        ax.set_ylabel("Pipeline Index")

        # Colorbar
        fig.colorbar(cax, ax=ax, label="Normalized LCS Similarity")

        plt.tight_layout()
        plt.show()

            
        # ----------------------------
        # 1. Find frequent subsequences
        # ----------------------------
        def extract_subsequences(seq, min_len=2, max_len=None):
            if max_len is None:
                max_len = len(seq)
            for length in range(min_len, max_len + 1):
                for indices in combinations(range(len(seq)), length):
                    yield tuple(seq[i] for i in indices)

        def find_frequent_subsequences(sequences, min_len=3, min_support=3):
            subseq_counts = defaultdict(set)
            for i, seq in enumerate(sequences):
                seen = set()
                for subseq in extract_subsequences(seq, min_len=min_len):
                    if subseq not in seen:
                        subseq_counts[subseq].add(i)
                        seen.add(subseq)
            return {
                subseq: idxs
                for subseq, idxs in subseq_counts.items()
                if len(idxs) >= min_support
            }

        frequent = find_frequent_subsequences(pipelines, min_len=4, min_support=3)

        # ----------------------------
        # 2. Build bipartite graph
        # ----------------------------
        B = nx.Graph()
        for i in range(len(pipelines)):
            B.add_node(f"S{i}", bipartite=0)

        for subseq, seq_ids in frequent.items():
            label = f"{subseq}"
            B.add_node(label, bipartite=1)
            for i in seq_ids:
                B.add_edge(f"S{i}", label)

        # ----------------------------
        # 3. Visualize the bipartite graph
        # ----------------------------
        # Separate nodes by type
        sequence_nodes = [n for n in B.nodes if n.startswith("S")]
        subsequence_nodes = list(set(B.nodes) - set(sequence_nodes))

        # Position nodes using bipartite layout
        pos = {}
        pos.update((n, (0, i)) for i, n in enumerate(sequence_nodes))
        pos.update((n, (1, i)) for i, n in enumerate(subsequence_nodes))

        # Draw
        plt.figure(figsize=(12, 8))
        nx.draw(B, pos, with_labels=True, node_size=1000, node_color="#f0f0f0", edge_color="gray", font_size=9)
        plt.title("Shared Subsequences Across Pipelines (min length 3, support ≥ 3)")
        plt.axis("off")
        plt.show()

        # Rank frequent subsequences by support and length
        ranked_subsequences = sorted(
            frequent.items(),
            key=lambda x: (-len(x[1]), -len(x[0]))
        )

        print("Top-ranked shared subsequences:")
        for subseq, seqs in ranked_subsequences[:10]:
            print(f"{subseq} → used in {len(seqs)} sequences: {sorted(seqs)}")

        # Build vocabulary of subsequences
        vocab = list(frequent.keys())

        # Create binary matrix: [pipeline][subsequence] = 1 if used
        binary_matrix = np.zeros((len(pipelines), len(vocab)))
        for j, subseq in enumerate(vocab):
            for i in frequent[subseq]:
                binary_matrix[i, j] = 1

        # Compute Jaccard distances
        distances = pairwise_distances(binary_matrix, metric="jaccard")

        # Agglomerative clustering
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='precomputed', linkage='average')
        labels = model.fit_predict(distances)

        # Show result
        for i, label in enumerate(labels):
            print(f"Pipeline {i} → Cluster {label}")


        linkage = sch.linkage(distances, method='average')
        plt.figure(figsize=(10, 6))
        sch.dendrogram(linkage, labels=[f"S{i}" for i in range(len(pipelines))])
        plt.title("Pipeline Clustering by Shared Subsequences")
        plt.ylabel("Distance (1 - Jaccard)")
        plt.tight_layout()
        plt.show()

        
    def unprocessed_check(self, optimiser, a, b, c, d):
        errors = optimiser.is_active(optimiser.snirf_data, print_result=True)

        print(errors)

        odds_ratio, p = fisher_exact([[a,b],[c,d]], alternative="greater")

        print(f"Odds ratio: {odds_ratio}")
        print(f"P-value: {p}")

        # Haldane-Anscombe correction (add 0.5 to each cell to avoid zeros)
        a_corr = a + 0.5
        b_corr = b + 0.5
        c_corr = c + 0.5
        d_corr = d + 0.5

        OR_corrected = (a_corr * d_corr) / (b_corr * c_corr)
        print(f"Corrected odds ratio: {OR_corrected:.2f}")

        # Calculate standard error of log(OR)
        se_log_or = np.sqrt(1/a_corr + 1/b_corr + 1/c_corr + 1/d_corr)

        # Calculate 95% confidence interval for corrected OR
        log_or = np.log(OR_corrected)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
        print(f"95% CI for corrected OR: [{ci_lower:.2f}, {ci_upper:.2f}]")

    def krippendorff_comparison(self):
        filepath = "interrater_KrippendorffsAlpha.xlsx"
        df = pd.read_excel(filepath, header=None, usecols="C:AM", skiprows=2, nrows=10)

        # Replace missing values with np.nan if needed (e.g., blanks)
        data = df.replace(r'^\s*$', np.nan, regex=True)

        # Map string ratings to integers
        value_map = {
            'Yes': 1,
            'No': 0,
            'Not investigated': 2
        }

        encoded = data.applymap(lambda x: value_map.get(x, np.nan)).to_numpy()

        # Transpose to (raters, items)
        data_t = encoded.T

        # Calculate Krippendorff's alpha (nominal)
        alpha = krippendorff.alpha(reliability_data=data_t, level_of_measurement='nominal')
        print(f"Krippendorff's alpha (nominal): {alpha:.3f}")

        # ==== Optimisation results ====

        df = pd.read_excel(filepath, header=None, usecols="C:L", skiprows=16, nrows=10)

        # Replace missing values with np.nan if needed (e.g., blanks)
        data = df.replace(r'^\s*$', np.nan, regex=True)

        # Map string ratings to integers
        value_map = {
            'Yes': 1,
            'No': 0,
        }

        encoded = data.applymap(lambda x: value_map.get(x, np.nan)).to_numpy()

        # Transpose to (raters, items)
        data_t = encoded.T

        # Calculate Krippendorff's alpha (nominal)
        alpha = krippendorff.alpha(reliability_data=data_t, level_of_measurement='nominal')
        print(f"Krippendorff's alpha (nominal): {alpha:.3f}")
