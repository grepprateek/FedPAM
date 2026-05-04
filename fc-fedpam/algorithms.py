import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import expm
from scipy.special import logsumexp
import torch
import torch.optim as optim

from pgmpy.estimators import HillClimbSearch, ExpertKnowledge, BayesianEstimator, BIC
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import os
import json
from collections import Counter
from tqdm import tqdm

import logging
import warnings
from tqdm import tqdm
from functools import partialmethod

logging.getLogger("pgmpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

class Client():
    def learn_local_structure(self, dataset, target, max_iter=100):
        forbidden_edges = [(target, var) for var in dataset.columns if var != target]
        expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)
        
        
        structure = HillClimbSearch(dataset).estimate(
            scoring_method='bic-d', 
            expert_knowledge=expert_knowledge,
            max_iter=100
        )
        model = structure.fit(dataset, estimator=BayesianEstimator, prior_type='BDeu')
        return model
    
    def visualize_network(self, graph_or_model, filename='bayesian_network.png', target='Target', edge_weights=None):
        
        if hasattr(graph_or_model, 'edges'):
            G = nx.DiGraph()
            G.add_edges_from(graph_or_model.edges())
        else:
            G = graph_or_model

        plt.figure(figsize = (20, 20))
        pos = nx.spring_layout(G, k = 1.5, seed = 23)

        
        simple_nodes = [node for node in G.nodes() if node != target]
        has_target = target in G.nodes()

        if simple_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                nodelist = simple_nodes,
                                node_size = 2000,
                                node_color = '#f6aa22',
                                alpha = 0.7,
                                edgecolors = 'black',
                                linewidths = 2)
            
            nx.draw_networkx_labels(G, pos,
                                    labels = {n: n for n in simple_nodes},
                                    font_size = 15)
        
        if has_target:
            nx.draw_networkx_nodes(G, pos, 
                                nodelist = [target],
                                node_size = 2500,
                                node_color = '#2BA6A6',
                                alpha = 0.7,
                                edgecolors = 'black',
                                linewidths = 2)
            
            nx.draw_networkx_labels(G, pos,
                                    labels = {target: target},
                                    font_size = 18,
                                    font_weight = 'bold')
        
        if edge_weights is not None:
            
            if isinstance(edge_weights, pd.DataFrame):
                weight_dict = {(i, j): edge_weights.loc[i, j] 
                              for i, j in G.edges() 
                              if i in edge_weights.index and j in edge_weights.columns}
            else:
                weight_dict = edge_weights
               
            edge_widths = []
            edge_labels = {}
            for (u, v) in G.edges():
                weight = weight_dict.get((u, v), 0.5)
                edge_widths.append(weight * 5)  
                edge_labels[(u, v)] = f'{weight:.2f}'
              
            nx.draw_networkx_edges(G, pos,
                                edgelist = G.edges(),
                                width = edge_widths,
                                arrowsize = 25,
                                alpha = 0.6,
                                connectionstyle = 'arc3, rad = 0.1')
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                        font_size = 10,
                                        bbox = dict(boxstyle='round,pad=0.3', 
                                                   facecolor='white', 
                                                   edgecolor='gray',
                                                   alpha=0.7))
        else:
            nx.draw_networkx_edges(G, pos,
                                edgelist = G.edges(),
                                arrowsize = 25,
                                connectionstyle = 'arc3, rad = 0.1')
        
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        title = F"Bayesian Network\nNodes: {num_nodes} | Edges: {num_edges}"
        if edge_weights is not None:
            title += " | Weighted"
        
        plt.title(title, fontsize = 25, pad = 20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  

    def _count_multilogit_params(self, dag, dataset):
        total = 0
        for node in dag.nodes():
            parents = list(dag.predecessors(node))
            n_classes = dataset[node].nunique()
            if len(parents) == 0:
                total += n_classes - 1
            else:
                total += (n_classes - 1) * (len(parents) + 1)

        return total

    def create_prob_adj_matrix(self, dataset, target, num_iterations=5, min_iterations=5, patience=5):
        num_samples = len(dataset)
        all_edges = []
        
        
        bic_scores = []
        best_avg_bic = -float('inf')
        iterations_without_improvement = 0
        
        print(f"Starting bootstrap with max_iterations={num_iterations}, min_iterations={min_iterations}, patience={patience}")
        
        for i in tqdm(range(num_iterations), desc="Bootstrapping Progress:\n"):
            bootstrap_sample = dataset.sample(n=num_samples, replace=True)
            bootstrap_model = self.learn_local_structure(bootstrap_sample, target)
            all_edges.extend(bootstrap_model.edges())
            
            
            if i >= min_iterations - 1:
                bic_scorer = BIC(bootstrap_sample)
                total_bic = bic_scorer.score(bootstrap_model)
                bic_scores.append(total_bic)
                current_avg_bic = np.mean(bic_scores)
                
                print(f"\n  Bootstrap {i+1}: Total BIC = {total_bic:.2f}, Avg BIC = {current_avg_bic:.2f}")
                
                if current_avg_bic > best_avg_bic:
                    improvement = current_avg_bic - best_avg_bic
                    best_avg_bic = current_avg_bic
                    iterations_without_improvement = 0
                    print(f"  New best Avg BIC! Improvement: {improvement:.2f}")
                else:
                    iterations_without_improvement += 1
                    print(f"  No improvement in Avg BIC ({iterations_without_improvement}/{patience})")
                
                
                if iterations_without_improvement >= patience:
                    print(f"\n  Early stopping triggered after {i+1} iterations")
                    print(f"  Avg BIC not improved for {patience} consecutive iterations")
                    print(f"  Best Avg BIC: {best_avg_bic:.2f}")
                    break
        
        final_iterations = i + 1
        print(f"\nBootstrap completed: {final_iterations} iterations (requested: {num_iterations})")
        if final_iterations < num_iterations:
            print(f"Saved {num_iterations - final_iterations} iterations via early stopping!")
        
        counts = Counter(all_edges)
        edge_strengths = {edge: count / final_iterations for edge, count in counts.items()}
        nodes = sorted(dataset.columns.tolist())

        prob_adj_matrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
        for (u, v), strength in edge_strengths.items():
            prob_adj_matrix.at[u, v] = strength

        return prob_adj_matrix, edge_strengths
    
    def visualize_pam(self, prob_adj_matrix, filename = 'prob_adj_matrix.png'):
        plt.figure(figsize=(20, 20))
        sns.heatmap(prob_adj_matrix, annot=True, cmap='Blues', cbar_kws={'label': 'Edge Strength'})
        plt.title("Probabilistic Adjacency Matrix")
        plt.xlabel("Child")
        plt.ylabel("Parent")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def visualize_metrics_over_iterations(self, metrics_history, filename='metrics_evolution.png'):
        iterations = metrics_history['iteration']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        
        ax1 = axes[0, 0]
        ax1.plot(iterations, metrics_history['bic'], 'o-', color='#2BA6A6', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('BIC Score', fontsize=12)
        ax1.set_title('BIC Evolution (Higher is Better)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        acc_mean = metrics_history['accuracy_mean']
        acc_std = metrics_history['accuracy_std']
        ax2.plot(iterations, acc_mean, 'o-', color='#f6aa22', linewidth=2, markersize=8)
        ax2.fill_between(iterations, 
                         np.array(acc_mean) - np.array(acc_std),
                         np.array(acc_mean) + np.array(acc_std),
                         alpha=0.2, color='#f6aa22')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Classification Accuracy (5-Fold CV)', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        
        ax3 = axes[1, 0]
        auroc_mean = metrics_history['auroc_mean']
        auroc_std = metrics_history['auroc_std']
        ax3.plot(iterations, auroc_mean, 'o-', color='#9b59b6', linewidth=2, markersize=8)
        ax3.fill_between(iterations,
                         np.array(auroc_mean) - np.array(auroc_std),
                         np.array(auroc_mean) + np.array(auroc_std),
                         alpha=0.2, color='#9b59b6')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('AUROC-OVR', fontsize=12)
        ax3.set_title('AUROC One-vs-Rest (5-Fold CV)', fontsize=14, fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)       
        ax4 = axes[1, 1]
        ax4.plot(iterations, metrics_history['edges'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Number of Edges', fontsize=12)
        ax4.set_title('Network Complexity', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics visualization saved to {filename}")

    def visualize_binary_matrix(self, binary_matrix, threshold, filename = 'binary_matrix.png'):
        plt.figure(figsize=(20, 20))
        sns.heatmap(binary_matrix, annot=True, cmap="Greys", cbar=False)
        plt.title(f"Binarized Adjacency Matrix (Li's Threshold: {threshold:.2f})", )
        plt.xlabel("Child")
        plt.ylabel("Parent")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)

    def create_dag(self, binary_matrix, edge_strengths, max_cycle_breaks=1000, verbose=False):
        G = nx.from_pandas_adjacency(binary_matrix, create_using = nx.DiGraph)
        if verbose:
            print(f"Initial edge count: {G.number_of_edges()}")

        edges_to_remove = []
        for u, v in G.edges():
            if G.has_edge(v, u):  
                
                if edge_strengths.get((u, v), 0) >= edge_strengths.get((v, u), 0):
                    edges_to_remove.append((v, u)) 
                else:
                    edges_to_remove.append((u, v))

        edges_to_remove = list(set(edges_to_remove))
        G.remove_edges_from(edges_to_remove)
        if verbose:
            print(f"Removed {len(edges_to_remove)} reciprocal edges.")

        cycle_breaks = 0
        while not nx.is_directed_acyclic_graph(G):
            if cycle_breaks >= max_cycle_breaks:
                print(f"WARNING: Reached maximum cycle breaks ({max_cycle_breaks}). Stopping to prevent infinite loop.")
                print(f"Graph still has cycles remaining.")
                break
                
            cycle = nx.find_cycle(G, orientation='original')
            weakest_edge = min(cycle, key=lambda e: edge_strengths.get((e[0], e[1]), 0))
            
            G.remove_edge(weakest_edge[0], weakest_edge[1])
            if verbose and cycle_breaks % 50 == 0:
                print(f"Breaking cycle {cycle_breaks}: removed {weakest_edge[0]} -> {weakest_edge[1]}")
            cycle_breaks += 1

        if verbose:
            print(f"{'Final Structure is a valid DAG.' if nx.is_directed_acyclic_graph(G) else 'Cycles still exist!'}")
        return G

    def prune_dag(self, dataset, dag, max_iterations=10, tol_bic=10):
        dag = {node: list(parents) for node, parents in dag.items()}
        fitted_models = {}
        cache = {}

        for i in range(max_iterations):
            changed = False
            edges_before = sum(len(v) for v in dag.values())

            for node in dag:
                parents = list(dag[node])
                bic_current, model_current = self.get_bic_model(
                    dataset=dataset,
                    node=node,
                    parents=parents,
                    cache=cache,
                )

                improved = True
                while improved and len(parents) > 0:
                    improved = False
                    best_bic = bic_current
                    best_remove = None
                    best_model = model_current

                    for p in parents:
                        candidate = [q for q in parents if q != p]

                        bic_candidate, model_candidate = self.get_bic_model(
                            dataset=dataset,
                            node=node,
                            parents=candidate,
                            cache=cache
                        )

                        if bic_candidate < best_bic - tol_bic:
                            best_bic = bic_candidate
                            best_remove = p
                            best_model = model_candidate

                    if best_remove is not None:
                        parents.remove(best_remove)
                        bic_current = best_bic
                        model_current = best_model
                        improved = True
                        changed = True
                        print(f"[Iteration {i + 1}] Node: {node} | Parent removed: {best_remove} -> BIC = {bic_current:.3f}")

                dag[node] = parents
                fitted_models[node] = model_current

            edges_after = sum(len(v) for v in dag.values())
            print(f"[Iteration {i + 1}] Edges: {edges_before} -> {edges_after}")

            if not changed:
                break

        return dag, fitted_models

    def pam_to_edge_strengths(self, pam: pd.DataFrame, eps: float = 0.0):
        strengths = {}
        for u in pam.index:
            for v in pam.columns:
                if u == v:
                    continue
                s = float(pam.loc[u, v])
                if s > eps:
                    strengths[(u, v)] = s
        return strengths
    
    def networkx_to_dag_dict(self, G):
        dag = {node: [] for node in G.nodes()}
        for parent, child in G.edges():
            dag[child].append(parent)
        return dag

    def dag_to_networkx(self, dag):
        G = nx.DiGraph()
        G.add_nodes_from(dag.keys())        
        for child, parents in dag.items():
            for p in parents:
                G.add_edge(p, child)
        return G

    def pairwise_cmi(self, xi_vals, xj_vals, levels_i, levels_j, laplace=1.0, eps=1e-12):
        idx_i = {v: a for a, v in enumerate(levels_i)}
        idx_j = {v: b for b, v in enumerate(levels_j)}

        counts = np.zeros((len(levels_i), len(levels_j)), float)
        for a, b in zip(xi_vals, xj_vals):
            counts[idx_i[a], idx_j[b]] += 1
        counts += laplace

        p_ij = counts / counts.sum()
        p_i  = p_ij.sum(axis=1, keepdims=True)
        p_j  = p_ij.sum(axis=0, keepdims=True)

        mi  = np.sum(p_ij * (np.log(p_ij + eps) - np.log(p_i + eps) - np.log(p_j + eps)))
        h_j = -np.sum(p_j * np.log(p_j + eps))

        return float(max(mi / max(h_j, eps), 0.0))

    
    def get_markov_blanket(self, pam, xj, threshold=0.1, max_cond=4, verbose=False):
        nodes = list(pam.columns)

        parents   = [xk for xk in nodes if xk != xj and pam.loc[xk, xj] > threshold]
        children  = [xk for xk in nodes if xk != xj and pam.loc[xj, xk] > threshold]
        co_parents = [
            xk for child in children
            for xk in nodes
            if xk != xj and xk not in parents and pam.loc[xk, child] > threshold
        ]

        mb = list(dict.fromkeys(parents + children + co_parents))  

        if verbose:
            print(f"  MB({xj}): parents={parents}, children={children}, co_parents={co_parents} -> MB={mb}")

        
        if len(mb) > max_cond:
            mb = sorted(mb, key=lambda v: pam.loc[v, xj] + pam.loc[xj, v], reverse=True)[:max_cond]
            if verbose:
                print(f"  MB({xj}) capped to {max_cond}: {mb}")

        return mb
    
    def cmi_given_cond_set(self, df, xi, xj, cond_set, laplace=1.0, eps=1e-12):
        nodes  = [xi, xj] + cond_set
        levels = {c: np.sort(df[c].unique()) for c in nodes}

        if len(cond_set) == 0:
            return self.pairwise_cmi(
                df[xi].values, df[xj].values,
                levels[xi], levels[xj],
                laplace, eps
            )

        idx_i = {v: a for a, v in enumerate(levels[xi])}
        idx_j = {v: b for b, v in enumerate(levels[xj])}
        idx_s = [{v: c for c, v in enumerate(levels[s])} for s in cond_set]

        shape  = (len(levels[xi]), len(levels[xj])) + tuple(len(levels[s]) for s in cond_set)
        counts = np.zeros(shape, float)

        cols_S = [df[s].values for s in cond_set]
        for row in zip(df[xi].values, df[xj].values, *cols_S):
            a     = idx_i[row[0]]
            b     = idx_j[row[1]]
            s_idx = tuple(idx_s[t][row[2 + t]] for t in range(len(cond_set)))
            counts[(a, b) + s_idx] += 1

        counts += laplace
        p = counts / counts.sum()

        p_s      = p.sum(axis=(0, 1))
        p_xi_s   = p.sum(axis=1)
        p_xj_s   = p.sum(axis=0)

        ratio = (p * (p_s[None, None, ...] + eps)) / (
            (p_xi_s[:, None, ...] + eps) * (p_xj_s[None, :, ...] + eps)
        )
        cmi_val = np.sum(p * np.log(ratio + eps))

        
        p_xj_given_s = (p_xj_s + eps) / (p_s[None, ...] + eps)
        h_xj_given_s = -np.sum(p_xj_s * np.log(p_xj_given_s + eps))

        return float(max(cmi_val / max(h_xj_given_s, eps), 0.0))
    
    def compute_cmi_reward_matrix(self, dataset, pam, threshold=0.1, max_cond=4, laplace=1.0, eps=1e-12, verbose=False):
        nodes = list(dataset.columns)
        d     = len(nodes)
        mat   = np.zeros((d, d))

        if verbose:
            print("Computing CMI reward matrix...")
            print(f"Nodes: {nodes}, threshold: {threshold}, max_cond: {max_cond}")

        for b, xj in enumerate(nodes):
            if verbose:
                print(f"Processing target node: {xj}")

            mb = self.get_markov_blanket(pam, xj, threshold, max_cond, verbose=verbose)

            for a, xi in enumerate(nodes):
                if xi == xj:
                    continue

                cond_set = [v for v in mb if v != xi]

                score = self.cmi_given_cond_set(dataset, xi, xj, cond_set, laplace, eps)
                mat[a, b] = score

                if verbose:
                    print(f"  C[{xi}->{xj}] | cond={cond_set} = {score:.4f}")

        np.fill_diagonal(mat, 0.0)
        C = pd.DataFrame(mat, index=nodes, columns=nodes)
        return C
    
    def optimize_pam(self, P_local, P_global, C, mu=0.1, lam=0.1):
        P_star = (2 * P_local.values + mu * P_global.values + lam * C.values) / (2 + mu + lam)
        P_star = np.clip(P_star, 0.0, 1.0)
        np.fill_diagonal(P_star, 0.0)
        return pd.DataFrame(P_star, index=P_local.index, columns=P_local.columns)

    def get_bic_model(self, dataset, node, parents, cache):
        key = (node, tuple(sorted(parents)))
        if key in cache:
            return cache[key]
        
        bic        = BIC(dataset)
        bic_score  = bic.local_score(node, parents)
        cache[key] = (bic_score, None)
        return bic_score, None
    
    def _count_multilogit_params(self, dag, dataset):
        total_params = 0
        
        for node in dag.nodes():
            r_j = dataset[node].nunique()  
            parents = list(dag.predecessors(node))
            
            if len(parents) == 0:
                total_params += (r_j - 1)
            else:
                parent_dims = sum(dataset[p].nunique() - 1 for p in parents)
                total_params += (r_j - 1) * (1 + parent_dims)
        
        return total_params

    
    def bic_threshold_search(self, P_star, dataset):
        
        candidates = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        best_bic = -float('inf')
        best_dag = None
        best_tau = None
        best_params = None
        
        edge_strengths = self.pam_to_edge_strengths(P_star, eps=0.0)
        
        for tau in candidates:
            
            binary_pam = (P_star >= tau).astype(int)
            np.fill_diagonal(binary_pam.values, 0)    
            dag = self.create_dag(binary_pam, edge_strengths, verbose=False)
            bic_scorer = BIC(dataset)
            bic = bic_scorer.score(dag)
            
            if bic > best_bic:
                best_bic = bic
                best_dag = dag
                best_tau = tau
                best_params = None
        
        return best_dag, best_tau, best_bic, best_params
    
    def pam_to_sparse(self, pam, threshold=1e-10):
        nodes = list(pam.index)
        edges = []
        
        for i in nodes:
            for j in nodes:
                value = pam.loc[i, j]
                if abs(value) > threshold:
                    edges.append((i, j, float(value)))
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def sparse_to_pam(self, sparse_data):
        nodes = sparse_data['nodes']
        edges = sparse_data['edges']
        pam = pd.DataFrame(0.0, index=nodes, columns=nodes)
        
        for (source, target, weight) in edges:
            if source in pam.index and target in pam.columns:
                pam.loc[source, target] = weight
        
        return pam

    def encode_dummy_vars(self, X):
        num_observations = len(X)
        num_states = len(np.unique(X))
        num_dummy_vars = num_states - 1
        dummy_vars = np.zeros((num_observations, num_dummy_vars))
        for level in range(num_dummy_vars):
            dummy_vars[:, level] = (X == level + 1 ).astype(float)
        return dummy_vars
    
    def create_design_matrix(self, dataset):
        num_observations, num_variables = dataset.shape
        encoded_parts = []
        col_ranges = {} 
        current_col = 1 

        for idx, variable in enumerate(dataset):
            num_states_i = len(np.unique(dataset.iloc[:, idx]))
            num_dummy_vars_i = num_states_i - 1
            dummy_vars_i = self.encode_dummy_vars(dataset.iloc[:, idx])
            encoded_parts.append(dummy_vars_i)
            col_ranges[variable] = slice(current_col, current_col + num_dummy_vars_i)
            current_col += num_dummy_vars_i
        
        intercept_col = np.ones((num_observations, 1))
        X_mat = np.hstack([intercept_col] + encoded_parts)
        return X_mat, col_ranges

    def softmax(self, eta):
        eta_shifted = eta - eta.max(axis=1, keepdims=True) 
        exp_eta = np.exp(eta_shifted)
        return exp_eta / exp_eta.sum(axis=1, keepdims=True)

    def compute_probabilities(self, X_mat, beta_j):
        eta = X_mat @ beta_j.T 
        return self.softmax(eta)

    def log_likelihood_node(self, X_mat, Y_j, beta_j):
        num_observations = len(Y_j)
        probs = self.compute_probabilities(X_mat, beta_j)
        
        
        log_lik = 0.0
        for h in range(num_observations):
            l = int(Y_j[h]) - 1
            log_lik += np.log(probs[h, l])
        return log_lik

    def _estimate_marginal(self, series):
        value_counts = series.value_counts(normalize=True, sort=False)
        
        levels = sorted(series.unique())
        probs = np.array([value_counts.get(level, 0.0) for level in levels])
        return probs

    def estimate_multilogit_params(self, dag, dataset):

        if not isinstance(dag, nx.DiGraph):
            dag = nx.DiGraph(dag.edges())

        params = {}
        for node in dag.nodes():
            parents = list(dag.predecessors(node))

            if len(parents) == 0:
                classes = np.sort(dataset[node].unique())
                counts = dataset[node].value_counts(normalize=True)
                probs = np.array([counts.get(c,0) for c in classes])
                params[node] = {
                    "intercept": probs,
                    "classes": classes
                }
                continue
            
            X = dataset[parents].copy()
            encoders = {}
    
            for col in X.columns:
                if X[col].dtype == 'object' or not np.issubdtype(X[col].dtype, np.number):
                    
                    encoder = LabelEncoder()
                    X[col] = encoder.fit_transform(X[col].astype(str))
                    encoders[col] = encoder
            
            X_values = X.values
            y = dataset[node].values
            
            unique_classes = np.unique(y)
            if len(unique_classes) == 1:
                classes = np.sort(dataset[node].unique())  
                if len(classes) == 1:
                    probs = np.array([1.0])
                else:
                    counts = dataset[node].value_counts(normalize=True)
                    probs = np.array([counts.get(c, 1e-10) for c in classes])
                    probs = probs / probs.sum()  
                
                params[node] = {
                    "intercept": probs,
                    "classes": classes,
                    "parents": parents,
                    "encoders": encoders,
                    "single_class_fallback": True  
                }
                
                continue

            model = LogisticRegression(
                solver="lbfgs",
                max_iter=500
            )

            model.fit(X_values, y)

            params[node] = {
                "parents": parents,
                "coefficients": model.coef_,
                "intercept": model.intercept_,
                "classes": model.classes_,
                "n_classes": len(model.classes_),
                "encoders": encoders  
            }

        return params
    
    def _predict_node_proba(self, parent_data, node_params):

        if "coefficients" not in node_params:
            
            if "intercept" not in node_params:
                raise ValueError(f"Node params missing 'intercept': {node_params.keys()}")
            
            probs = node_params["intercept"]
            
            if isinstance(probs, (list, tuple, np.ndarray)):
                probs = np.asarray(probs)
                if probs.ndim == 1:
                    
                    probs = probs.reshape(1, -1)
                return np.tile(probs, (len(parent_data), 1))
            else:
                raise ValueError(f"Invalid intercept type: {type(probs)}")

        
        if not isinstance(parent_data, pd.DataFrame):
            parent_data = pd.DataFrame(parent_data, columns=node_params.get("parents", None))
        
        encoders = node_params.get("encoders", {})
        if encoders:
            parent_data = parent_data.copy()
            for col, encoder in encoders.items():
                if col in parent_data.columns:
                    parent_data[col] = encoder.transform(parent_data[col].astype(str))
        
        parent_data_values = parent_data.values
        coef = node_params["coefficients"]
        intercept = node_params.get("intercept", 0.0)
        logits = parent_data_values @ coef.T + intercept
        
        if not isinstance(logits, np.ndarray):
            logits = np.asarray(logits)

        if logits.shape[1] == 1:
            p1 = 1 / (1 + np.exp(-logits))
            p0 = 1 - p1
            probs = np.hstack([p0, p1])
            return probs

        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs
    
    def compute_multilogit_loglik(self, dag, params, dataset):
        log_likelihood = 0
        n_samples = len(dataset)

        for node, node_params in params.items():
            y_true = dataset[node].values
            if "parents" in node_params:
                parents = node_params["parents"]
                parent_data = dataset[parents]
                if parent_data.ndim == 1:
                    parent_data = parent_data.to_frame()
                probs = self._predict_node_proba(parent_data, node_params)
            else:
                probs = np.tile(node_params["intercept"], (n_samples,1))

            classes = node_params["classes"]
            class_to_idx = {c:i for i,c in enumerate(classes)}
            y_idx = np.array([class_to_idx[v] for v in y_true])
            probs = np.clip(probs, 1e-10, 1-1e-10)
            log_likelihood += np.sum(
                np.log(probs[np.arange(n_samples), y_idx])
            )

        return log_likelihood
    
    def predict_target(self, dag, params, dataset, target='Target'):
        if target not in dag.nodes():
            raise ValueError(f"Target {target} not in DAG")
        
        
        parents = list(dag.predecessors(target))
        
        if len(parents) == 0:
            
            probs = params[target]['intercept']
            n_samples = len(dataset)
            y_proba = np.tile(probs, (n_samples, 1))
        else:
            
            parent_data = dataset[parents]
            y_proba = self._predict_node_proba(parent_data, params[target])
        
        
        y_pred = np.argmax(y_proba, axis=1)
        
        return y_pred, y_proba
    
    def save_predictions_to_csv(self, dag, dataset, output_path, target='Target', params=None):
        df_with_predictions = dataset.copy()
        if params is None:
            params = self.estimate_multilogit_params(dag, dataset)
        
        y_pred, y_proba = self.predict_target(dag, params, dataset, target)
        
        levels = sorted(dataset[target].unique())
        
        pred_col = f'Predicted_{target}'
        df_with_predictions[pred_col] = [levels[idx] for idx in y_pred]
        
        for idx, level in enumerate(levels):
            df_with_predictions[f'Probability_{level}'] = y_proba[:, idx]

        df_with_predictions['Prediction_Confidence'] = y_proba.max(axis=1)
        df_with_predictions.to_csv(output_path, index=False)
        
        print(f"Predictions saved to: {output_path}")
        print(f"  - Added column: '{pred_col}' (predicted class)")
        print(f"  - Added columns: 'Probability_<class>' for each class")
        print(f"  - Added column: 'Prediction_Confidence' (max probability)")
        
        return df_with_predictions
    
    def evaluate_kfold_cv(self, dag, dataset, target='Target', k=5, random_state=23):
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        
        fold_results = []
        accuracies = []
        aurocs = []
        f1_scores = []  
        
        
        y_true_all = dataset[target].values
        levels = sorted(dataset[target].unique())
        n_classes = len(levels)
        level_to_idx = {level: idx for idx, level in enumerate(levels)}
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
            train_data = dataset.iloc[train_idx]
            test_data = dataset.iloc[test_idx]

            fold_params = self.estimate_multilogit_params(dag, train_data)            
            y_pred, y_proba = self.predict_target(dag, fold_params, test_data, target)
            y_true_fold = test_data[target].values
            y_true_idx = np.array([level_to_idx[val] for val in y_true_fold])
            
            accuracy = accuracy_score(y_true_idx, y_pred)
            accuracies.append(accuracy)
                       
            f1 = f1_score(y_true_idx, y_pred, average='macro', zero_division=0)
            f1_scores.append(f1)
            
            try:
                unique_classes_in_fold = np.unique(y_true_idx)
                if n_classes == 2:
                    if len(unique_classes_in_fold) == 2:
                        auroc = roc_auc_score(y_true_idx, y_proba[:, 1])
                    else:
                        auroc = None
                        print(f"  WARNING: Fold {fold_idx+1} has only one class, skipping AUROC")
                else:
                    if len(unique_classes_in_fold) == n_classes:   
                        auroc = roc_auc_score(y_true_idx, y_proba, 
                                             multi_class='ovr', 
                                             average='weighted')
                    else:
                        auroc = roc_auc_score(y_true_idx, y_proba,
                                             multi_class='ovr',
                                             average='weighted',
                                             labels=list(range(n_classes)))
                
                if auroc is not None:
                    aurocs.append(auroc)
                    
            except Exception as e:
                print(f"  WARNING: Could not compute AUROC for fold {fold_idx+1}: {e}")
                auroc = None
            
            fold_results.append({
                'fold': fold_idx + 1,
                'accuracy': accuracy,
                'f1_score': f1,  
                'auroc': auroc,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'n_classes_in_fold': len(unique_classes_in_fold)
            })
        
        
        if len(aurocs) > 0:
            auroc_mean = np.mean(aurocs)
            auroc_std = np.std(aurocs)
        else:
            
            auroc_mean = np.nan
            auroc_std = np.nan
            print("  WARNING: Could not compute AUROC for any fold")
        
        return {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'f1_mean': np.mean(f1_scores),  
            'f1_std': np.std(f1_scores),    
            'auroc_mean': auroc_mean,
            'auroc_std': auroc_std,
            'fold_results': fold_results
        }
    
    
class Coordinator(Client):    
    def align_pams(self, pams_list):
        
        all_nodes = set()
        for pam in pams_list:
            all_nodes.update(pam.index)
        
        all_nodes = sorted(list(all_nodes))
        
        aligned = []
        for pam in pams_list:
            aligned_pam = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
        
            for i in pam.index:
                for j in pam.columns:
                    if i in aligned_pam.index and j in aligned_pam.columns:
                        aligned_pam.loc[i, j] = pam.loc[i, j]
            
            aligned.append(aligned_pam)
        
        return aligned
    
    def aggregate_pams(self, client_pams, client_weights=None, eps=1e-6):
        if len(client_pams) == 0:
            raise ValueError("client_pams is empty")

        aligned_pams = self.align_pams(client_pams)
        nodes = list(aligned_pams[0].index)

        K = len(aligned_pams)
        if client_weights is None:
            w = np.ones(K, dtype=float) / K
        else:
            w = np.asarray(client_weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("client_weights must be nonnegative")
            if np.sum(w) == 0:
                raise ValueError("Sum of client_weights is 0")
            w = w / np.sum(w)

        stacked = np.stack([pam.values.astype(float) for pam in aligned_pams], axis=0)  
        global_mat = np.tensordot(w, stacked, axes=(0, 0))  

        
        global_mat = np.clip(global_mat, eps, 1 - eps)
        np.fill_diagonal(global_mat, 0.0)

        return pd.DataFrame(global_mat, index=nodes, columns=nodes)
    
    def aggregate_betas(self, client_betas_list, client_weights):
        if len(client_betas_list) == 0:
            raise ValueError("client_betas_list is empty")
        
        
        w = np.asarray(client_weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("client_weights must be nonnegative")
        if np.sum(w) == 0:
            raise ValueError("Sum of client_weights is 0")
        w = w / np.sum(w)
        
        aggregated_betas = {}
        
        all_nodes = set(client_betas_list[0].keys())
        
        for node in all_nodes:
            
            node_params_list = []
            valid_weights = []
            
            for i, client_betas in enumerate(client_betas_list):
                if node in client_betas:
                    node_params_list.append(client_betas[node])
                    valid_weights.append(w[i])
            
            if len(node_params_list) == 0:
                continue
                        
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()
            first_params = node_params_list[0]
            
            if 'coefficients' in first_params:
                coef_list = []
                intercept_list = []
                
                for params in node_params_list:
                    if 'coefficients' not in params:
                        raise ValueError(f"Inconsistent parent structure for node '{node}': "
                                       f"Some clients have parents, others don't. "
                                       f"This suggests different DAG structures across clients.")
                    
                    coef_list.append(params['coefficients'])
                    intercept_list.append(params['intercept'])
        
                coef_shapes = [c.shape for c in coef_list]              
                n_classes_list = [shape[0] for shape in coef_shapes]
                n_features_list = [shape[1] if len(shape) > 1 else 1 for shape in coef_shapes]
                
                if len(set(n_features_list)) > 1:
                    raise ValueError(f"Inconsistent parent structure for node '{node}': {coef_shapes}. "
                                   "Different number of parent features suggests different DAG structures.")
                
                if len(set(n_classes_list)) > 1:
                    max_classes = max(n_classes_list)
                    n_features = n_features_list[0]
                    print(f"[COORDINATOR] Node '{node}': Aligning classes from {n_classes_list} to {max_classes}")
                    
                    aligned_coef_list = []
                    aligned_intercept_list = []
                    
                    for coef, intercept in zip(coef_list, intercept_list):
                        n_classes_current = coef.shape[0]
                        if n_classes_current < max_classes:
                            pad_size = max_classes - n_classes_current
                            
                            if coef.ndim == 1:
                                coef_padded = np.pad(coef, (0, pad_size), mode='constant', constant_values=0)
                            else:
                                coef_padded = np.pad(coef, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
                            
                            intercept_padded = np.pad(intercept, (0, pad_size), mode='constant', constant_values=0)
                            aligned_coef_list.append(coef_padded)
                            aligned_intercept_list.append(intercept_padded)
                        else:
                            aligned_coef_list.append(coef)
                            aligned_intercept_list.append(intercept)
                    
                    coef_list = aligned_coef_list
                    intercept_list = aligned_intercept_list
                
                coef_stacked = np.stack(coef_list, axis=0)  
                agg_coef = np.tensordot(valid_weights, coef_stacked, axes=(0, 0))
                
                intercept_stacked = np.stack(intercept_list, axis=0)  
                agg_intercept = np.tensordot(valid_weights, intercept_stacked, axes=(0, 0))
            
                aggregated_betas[node] = {
                    'coefficients': agg_coef,
                    'intercept': agg_intercept,
                    'parents': first_params.get('parents', []),
                    'classes': first_params.get('classes', None)
                }
            else:
                intercept_list = [params['intercept'] for params in node_params_list]               
                intercept_lengths = [len(ic) if hasattr(ic, '__len__') else 1 for ic in intercept_list]
                
                if len(set(intercept_lengths)) > 1:
                    max_classes = max(intercept_lengths)
                    print(f"[COORDINATOR] Node '{node}' (no parents): Aligning classes from {intercept_lengths} to {max_classes}")
                    
                    aligned_intercept_list = []
                    for intercept in intercept_list:
                        current_len = len(intercept) if hasattr(intercept, '__len__') else 1
                        if current_len < max_classes:
                            pad_size = max_classes - current_len
                            intercept_padded = np.pad(intercept, (0, pad_size), mode='constant', constant_values=0)
                            aligned_intercept_list.append(intercept_padded)
                        else:
                            aligned_intercept_list.append(intercept)
            
                    intercept_list = aligned_intercept_list
                
                intercept_stacked = np.stack(intercept_list, axis=0)
                agg_intercept = np.tensordot(valid_weights, intercept_stacked, axes=(0, 0))
                aggregated_betas[node] = {
                    'intercept': agg_intercept
                }
        
        return aggregated_betas
    
    def create_beta_ordering(self, dag_edges, dataset):
        """
        Create canonical ordering for beta parameters based on DAG structure.
        This ensures all clients flatten their betas in the same order.
        
        Returns:
            node_order: list of node names in topological order
            param_positions: dict mapping node -> {'intercept': (start, end), 'coefficients': (start, end)}
            total_params: total number of scalar parameters
        """
        import networkx as nx
        
        G = nx.DiGraph()
        G.add_edges_from(dag_edges or [])
        if dataset is not None:
            G.add_nodes_from(dataset.columns.tolist())
        
        if nx.is_directed_acyclic_graph(G):
            node_order = list(nx.topological_sort(G))
        else:
            node_order = sorted(list(G.nodes()))
        
        param_positions = {}
        current_pos = 0
        
        for node in node_order:
            param_positions[node] = {}
            
            if dataset is not None and node in dataset.columns:
                n_classes = dataset[node].nunique()
            else:
                n_classes = 2  
            
            parents = list(G.predecessors(node))
            n_parents = len(parents)
            
            intercept_size = n_classes
            param_positions[node]['intercept'] = (current_pos, current_pos + intercept_size)
            current_pos += intercept_size
            
            if n_parents > 0:
                coef_size = n_classes * n_parents
                param_positions[node]['coefficients'] = (current_pos, current_pos + coef_size)
                current_pos += coef_size
        
        total_params = current_pos
        
        return node_order, param_positions, total_params
    
    def create_beta_ordering_with_metadata(self, dag_edges, dataset):
        """
        Create canonical ordering for beta parameters AND extract metadata.
        This ensures all clients flatten their betas in the same order.
        
        Returns:
            node_order: list of node names in topological order
            param_positions: dict mapping node -> {'intercept': (start, end), 'coefficients': (start, end)}
            total_params: total number of scalar parameters
            beta_metadata: dict mapping node -> {'parents': list, 'classes': list}
        """
        import networkx as nx
        
        G = nx.DiGraph()
        G.add_edges_from(dag_edges or [])
        if dataset is not None:
            G.add_nodes_from(dataset.columns.tolist())
        
        if nx.is_directed_acyclic_graph(G):
            node_order = list(nx.topological_sort(G))
        else:
            node_order = sorted(list(G.nodes()))
        
        param_positions = {}
        beta_metadata = {}  
        current_pos = 0
        
        for node in node_order:
            param_positions[node] = {}
            parents = list(G.predecessors(node))
            
            if dataset is not None and node in dataset.columns:
                classes = sorted(dataset[node].unique())
                n_classes = len(classes)
            else:
                classes = [0, 1] 
                n_classes = 2
            
            beta_metadata[node] = {
                'parents': [str(p) for p in parents],
                'classes': [str(c) for c in classes]
            }
            
            intercept_size = n_classes
            param_positions[node]['intercept'] = (current_pos, current_pos + intercept_size)
            current_pos += intercept_size
            
            if len(parents) > 0:
                coef_size = n_classes * len(parents)
                param_positions[node]['coefficients'] = (current_pos, current_pos + coef_size)
                current_pos += coef_size
        
        total_params = current_pos
        
        return node_order, param_positions, total_params, beta_metadata
    
    def flatten_betas(self, betas_dict, node_order, param_positions, total_params):
        """
        Converts beta parameters dictionary to fixed-order vector.
        
        Args:
            betas_dict: dict with structure {node: {'intercept': array, 'coefficients': array, ...}}
            node_order: canonical node ordering
            param_positions: parameter position mapping
            total_params: total vector length
            
        Returns:
            flat_vector: numpy array of all parameters in canonical order
        """
        import numpy as np
        
        flat_vector = np.zeros(total_params)
        
        for node in node_order:
            if node not in betas_dict:
                continue
                
            node_params = betas_dict[node]
            
            if 'intercept' in node_params and 'intercept' in param_positions[node]:
                start, end = param_positions[node]['intercept']
                intercept = np.array(node_params['intercept']).flatten()
                expected_size = end - start
                if len(intercept) < expected_size:
                    intercept = np.pad(intercept, (0, expected_size - len(intercept)), mode='constant', constant_values=0)
                elif len(intercept) > expected_size:
                    intercept = intercept[:expected_size]
                flat_vector[start:end] = intercept
            
            if 'coefficients' in node_params and 'coefficients' in param_positions[node]:
                start, end = param_positions[node]['coefficients']
                coef = np.array(node_params['coefficients']).flatten()
                expected_size = end - start
                if len(coef) < expected_size:
                    coef = np.pad(coef, (0, expected_size - len(coef)), mode='constant', constant_values=0)
                elif len(coef) > expected_size:
                    coef = coef[:expected_size]
                flat_vector[start:end] = coef
        
        return flat_vector
    
    def unflatten_betas(self, flat_vector, node_order, param_positions, beta_metadata=None):
        """
        Convert flat vector back to beta parameters dictionary.
        
        Args:
            flat_vector: numpy array or list of parameters
            node_order: canonical node ordering
            param_positions: parameter position mapping
            beta_metadata: optional dict mapping node -> {'parents': list, 'classes': list}
            
        Returns:
            betas_dict: dict with structure {node: {'intercept': array, 'coefficients': array, 'parents': list, 'classes': array}}
        """
        import numpy as np
        
        aggregated_betas = {}
        
        for node in node_order:
            node_params = {}
            
            if 'intercept' in param_positions[node]:
                start, end = param_positions[node]['intercept']
                intercept = flat_vector[start:end]
                node_params['intercept'] = np.array(intercept)
            
            if 'coefficients' in param_positions[node]:
                start, end = param_positions[node]['coefficients']
                coef_flat = flat_vector[start:end]
                
                n_classes = len(node_params['intercept'])
                n_values = end - start
                n_parents = n_values // n_classes
                
                coef = np.array(coef_flat).reshape(n_classes, n_parents)
                node_params['coefficients'] = coef
            
            if beta_metadata is not None and node in beta_metadata:
                node_params['parents'] = beta_metadata[node]['parents']
                node_params['classes'] = np.array(beta_metadata[node]['classes'])
            
            aggregated_betas[node] = node_params
        
        return aggregated_betas