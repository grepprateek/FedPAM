from FeatureCloud.app.engine.app import AppState, Role, app_state, State
import time
import os
import yaml
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)

import warnings
import traceback
import json
import sys

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='pgmpy')
warnings.filterwarnings('ignore', message='.*Replacing existing CPD.*')
warnings.filterwarnings('ignore', message='.*pgmpy.*')

from algorithms import Client, Coordinator

INITIAL = 'initial'
READ_INPUT = 'read input'
LOCAL_COMPUTATION = 'local computation'
AGGREGATION = 'aggregation'
AWAIT_AGGREGATION = 'await aggregation'
FINAL = 'final'
TERMINAL = 'terminal'

def _json_default(o):  
    try:
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass

    try:
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="split")
        if isinstance(o, pd.Series):
            return o.to_list()
    except Exception:
        pass

    return str(o)

def _convert_tuple_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            
            if isinstance(k, tuple):
                new_key = str(k)
            else:
                new_key = k
            new_dict[new_key] = _convert_tuple_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [_convert_tuple_keys(item) for item in obj]
    else:
        return obj

def payload_size_bytes(payload: dict) -> int:
    
    payload_converted = _convert_tuple_keys(payload)
    s = json.dumps(payload_converted, default=_json_default, ensure_ascii=False, separators=(",", ":"))
    return len(s.encode("utf-8"))

@app_state(INITIAL, Role.BOTH)
class InitialState(AppState):
    def register(self):
        self.register_transition(READ_INPUT, Role.BOTH)

    def run(self):
        self.log("Starting FedPAM...")
        self.log(f"Node ID: {self.id}, Role: {'Coordinator' if self.is_coordinator else 'Client'}")
        
        
        self.store('start_time', time.time())
        self.store('client_payload_sizes', [])  
        self.store('coordinator_payload_sizes', [])  
        
        self.log("INITIAL to READ INPUT")
        return READ_INPUT


@app_state(READ_INPUT, Role.BOTH)
class ReadInputState(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION, Role.BOTH)
        self.register_transition(READ_INPUT, Role.BOTH)

    def read_config_file(self):
        self.log("Reading config file...")

        input_dir = "/mnt/input"
        output_dir = "/mnt/output"

        self.store('input_dir', input_dir)
        self.store('output_dir', output_dir)

        config_file_path = os.path.join(input_dir, 'config.yml')
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found at {config_file_path}")
        
        with open(config_file_path) as cfp:
            config_file = yaml.load(cfp, Loader = yaml.FullLoader)

        configs = config_file['fc-fedpam']
        self.store('dataset_loc', configs['input']['dataset_loc'])
        self.store('target', configs['input']['target'])
        self.store('split_mode', configs['split']['mode'])
        self.store('split_dir', configs['split']['dir'])
        self.store('max_iterations', configs['max_iterations'])
        self.store('fl_min_iterations', configs.get('fl_min_iterations', 5))
        self.store('fl_patience', configs.get('fl_patience', 5))
        self.store('bootstrap_iterations', configs['bootstrap_iterations'])
        self.store('bootstrap_min_iterations', configs.get('bootstrap_min_iterations', 5))
        self.store('bootstrap_patience', configs.get('bootstrap_patience', 5))
        self.store('mu', configs['mu'])
        self.store('lam', configs.get('lam', 0.1)) 
        self.store('homogeneous', configs['homogeneous']) 

        splits = {}
        if self.load('split_mode') == 'directory':
            split_base_dir = os.path.join(input_dir, self.load('split_dir'))
            if os.path.exists(split_base_dir):
                splits = {f.path: None for f in os.scandir(split_base_dir) if f.is_dir()}
            else:
                splits = {input_dir: None}
        else:
            splits = {input_dir: None}

        roles = {}
        for split_path in splits.keys():
            output_path = split_path.replace('/input/', '/output/')
            os.makedirs(output_path, exist_ok = True)

        self.log("Configuration Loaded Successfully!")
        self.store('splits', splits)
        self.store('roles', roles)
    
    def run(self):
        iteration = 1
        self.read_config_file()

        splits = self.load('splits')
        roles = self.load('roles')

        for split_path in splits.keys():
            roles[split_path] = 'coordinator' if self.is_coordinator else 'client'

            dataset_loc = self.load('dataset_loc')
            dataset_path = os.path.join(split_path, dataset_loc)
            if not os.path.exists(dataset_path):
                    raise FileNotFoundError(f"Dataset File not found at location: {dataset_path}")
                
            dataset = pd.read_csv(dataset_path)
            dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
            
            splits[split_path] = dataset
            self.log(f"[CLIENT] Loaded dataset from {split_path}: {dataset.shape[0]} records and {dataset.shape[1]} variables")
            self.log(f"[CLIENT] Dataset shuffled to prevent sorted Target issues")
            
            self.store('roles', roles)
            self.store('splits', splits)
            self.store('iteration', iteration)

            client_split_path = None
            client_id_str = str(self.id).lower()
            for split_path in splits.keys():
                split_dirname = os.path.basename(split_path).lower()
                self.log(f"[CLIENT] Comparing Client ID '{client_id_str}' with split directory '{split_dirname}'")
                if client_id_str in split_dirname: 
                    client_split_path = split_path
                    break 

            if client_split_path is None:
                if len(splits) == 1:
                    client_split_path = next(iter(splits.keys()))
                    self.log(f"[CLIENT] Using split directory: {client_split_path}")
                else:
                    raise RuntimeError(f"[CLIENT {self.id}]: No matching split directory found for this client.")

            self.store('dataset', splits[client_split_path])
            self.store('client_split_path', client_split_path)

            client = roles[client_split_path]
            self.store('client_instance', client)
            self.store('iteration', 1)

            self.log("READ INPUT TO LOCAL COMP")
            return LOCAL_COMPUTATION
    
@app_state(LOCAL_COMPUTATION, Role.BOTH)
class LocalComputationState(AppState):
    def register(self):
        self.register_transition(AGGREGATION, Role.COORDINATOR)
        self.register_transition(AWAIT_AGGREGATION, Role.PARTICIPANT)

    def run(self):
        output_dir = self.load('output_dir')
        iteration = self.load('iteration')
        dataset = self.load('dataset')
        dataset_size = len(dataset)
        target = self.load('target')
        mu = self.load('mu')
        homogeneous = self.load('homogeneous')
        self.log(f"ITERATION : {iteration}")

        participant = Coordinator() if self.is_coordinator else Client()

        if iteration == 1:
            number = np.random.randint(0,10,1)
            self.store('number', number)
            bootstrap_iterations = self.load('bootstrap_iterations')
            bootstrap_min_iterations = self.load('bootstrap_min_iterations')
            bootstrap_patience = self.load('bootstrap_patience')
            self.log(f"Using bootstrap_iterations={bootstrap_iterations}, min={bootstrap_min_iterations}, patience={bootstrap_patience}")
            
            local_pam, _ = participant.create_prob_adj_matrix(
                dataset, 
                target,
                num_iterations=bootstrap_iterations,
                min_iterations=bootstrap_min_iterations,
                patience=bootstrap_patience
            )
            self.store('local_pam', local_pam)
            local_cmi = participant.compute_cmi_reward_matrix(dataset, local_pam)
            self.store('local_cmi', local_cmi)
            
            
            best_dag, best_tau, best_bic, best_params = participant.bic_threshold_search(
                local_pam, dataset
            )
            
            self.log(f"[Iteration 1] BEST TAU: {best_tau}")
            self.log(f"[Iteration 1] BEST BIC: {best_bic}")
            self.log(f"[Iteration 1] EDGES: {best_dag.number_of_edges()}")
            
            
            self.log(f"[Iteration 1] Evaluating predictive performance...")
            cv_results = participant.evaluate_kfold_cv(
                best_dag, dataset, target=target, k=5, random_state=42
            )
            
            self.log(f"[Iteration 1] Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            self.log(f"[Iteration 1] F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
            self.log(f"[Iteration 1] AUROC-OVR: {cv_results['auroc_mean']:.4f} ± {cv_results['auroc_std']:.4f}")
            
            
            self.store('local_params', best_params)
            
            
            metrics_history = {
                'iteration': [1],
                'bic': [best_bic],
                'accuracy_mean': [cv_results['accuracy_mean']],
                'accuracy_std': [cv_results['accuracy_std']],
                'f1_mean': [cv_results['f1_mean']],  
                'f1_std': [cv_results['f1_std']],    
                'auroc_mean': [cv_results['auroc_mean']],
                'auroc_std': [cv_results['auroc_std']],
                'edges': [best_dag.number_of_edges()],
                'threshold': [best_tau]
            }
            self.store('metrics_history', metrics_history)
            
            participant.visualize_network(
                best_dag, 
                os.path.join(output_dir, f"local_dag_{iteration}.png"),
                target = target,
                edge_weights=local_pam
            )
            
        else:   
            number = self.load('number')
            global_sum = self.load('global_sum_local')
            number = number + global_sum

            local_pam = self.load('local_pam')  
            global_pam = self.load('global_pam_local')
            local_cmi = self.load('local_cmi')
            
            mu = self.load('mu')
            lam = self.load('lam')
            self.log(f"[Iteration {iteration}] Using mu={mu}, lam={lam} for PAM optimization")
            
            local_pam = participant.optimize_pam(local_pam, global_pam, local_cmi, mu=mu, lam=lam)
            self.store('local_pam', local_pam)  
            
            self.log(f"[Iteration {iteration}] Recomputing CMI with updated PAM...")
            local_cmi = participant.compute_cmi_reward_matrix(dataset, local_pam)
            self.store('local_cmi', local_cmi)  
            
            
            self.log(f"[Iteration {iteration}] Running BIC-based threshold search on refined local PAM...")
            local_dag, best_tau, best_bic, local_params = participant.bic_threshold_search(
                local_pam, dataset
            )
            self.store('local_dag_current', local_dag)
            self.store('local_params', local_params)
            
            
            global_beta_params = None
            if homogeneous:
                self.log(f"[Iteration {iteration}] HOMOGENEOUS MODE: "
                         f"using refined local PAM for local DAG and global DAG for beta aggregation")
                
                global_dag_edges = self.load('global_dag_edges_local')
                if not global_dag_edges or len(global_dag_edges) == 0:
                    raise RuntimeError(f"[Iteration {iteration}] HOMOGENEOUS MODE: No global DAG available!")
                
                global_dag = nx.DiGraph()
                global_dag.add_edges_from(global_dag_edges)
                self.log(f"[Iteration {iteration}] Global DAG: "
                         f"{global_dag.number_of_nodes()} nodes, {global_dag.number_of_edges()} edges")
                
                global_beta_params = participant.estimate_multilogit_params(global_dag, dataset)
            else:
                self.log(f"[Iteration {iteration}] HETEROGENEOUS MODE: "
                         f"using refined local PAM and local DAG for personalized parameters")
            
            best_dag = local_dag
            best_params = local_params
            
            self.log(f"[Iteration {iteration}] BEST TAU: {best_tau}")
            self.log(f"[Iteration {iteration}] BEST BIC: {best_bic}")
            self.log(f"[Iteration {iteration}] EDGES: {best_dag.number_of_edges()}")
            self.log(f"[Iteration {iteration}] NUM PARAMS: {participant._count_multilogit_params(best_dag, dataset)}")
            
            self.log(f"[Iteration {iteration}] BEST TAU: {best_tau}")
            self.log(f"[Iteration {iteration}] BEST BIC: {best_bic}")
            self.log(f"[Iteration {iteration}] EDGES: {best_dag.number_of_edges()}")
            self.log(f"[Iteration {iteration}] NUM PARAMS: {participant._count_multilogit_params(best_dag, dataset)}")
            
            
            self.log(f"[Iteration {iteration}] Evaluating predictive performance...")
            cv_results = participant.evaluate_kfold_cv(
                best_dag, dataset, target=target, k=5, random_state=42
            )
            
            self.log(f"[Iteration {iteration}] Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            self.log(f"[Iteration {iteration}] F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
            self.log(f"[Iteration {iteration}] AUROC-OVR: {cv_results['auroc_mean']:.4f} ± {cv_results['auroc_std']:.4f}")
            
            
            self.store('local_params', best_params)
            
            
            metrics_history = self.load('metrics_history')
            metrics_history['iteration'].append(iteration)
            metrics_history['bic'].append(best_bic)
            metrics_history['accuracy_mean'].append(cv_results['accuracy_mean'])
            metrics_history['accuracy_std'].append(cv_results['accuracy_std'])
            metrics_history['f1_mean'].append(cv_results['f1_mean'])  
            metrics_history['f1_std'].append(cv_results['f1_std'])    
            metrics_history['auroc_mean'].append(cv_results['auroc_mean'])
            metrics_history['auroc_std'].append(cv_results['auroc_std'])
            metrics_history['edges'].append(best_dag.number_of_edges())
            metrics_history['threshold'].append(best_tau)
            self.store('metrics_history', metrics_history)
            
            
            self.store('final_dag', best_dag)
            
            
            edge_weights = {}
            for (u, v) in best_dag.edges():
                if u in local_pam.index and v in local_pam.columns:
                    edge_weights[(u, v)] = float(local_pam.loc[u, v])
            
            participant.visualize_network(
                best_dag,
                os.path.join(output_dir, f"local_dag_{iteration}.png"),
                target=target,
                edge_weights=edge_weights
            )
            
            if homogeneous:
                self.log(f"[CLIENT] Homogeneous mode - Local DAG at Iteration {iteration} "
                         f"(weighted by refined local PAM): {best_dag.edges()}")
            else:
                self.log(f"[CLIENT] Heterogeneous mode - Local DAG at Iteration {iteration} "
                         f"(weighted by refined local PAM): {best_dag.edges()}")

            participant.visualize_pam(local_pam, os.path.join(output_dir, f"refined_local_pam_{iteration}.png"))

            if iteration > 1:
                global_pam_viz = self.load('global_pam_local')
                participant.visualize_pam(global_pam_viz, os.path.join(output_dir, f"global_pam_{iteration}.png"))


        participant_sparse_pam = participant.pam_to_sparse(local_pam)
        
        participant_payload = {
            "client_number": number,
            "client_pam_sparse": participant_sparse_pam,
            "client_dataset_size": dataset_size,
            "client_bic": best_bic,
            "client_threshold": best_tau
        }
        
        if homogeneous and iteration > 1 and global_beta_params is not None:
            participant_payload["client_betas"] = global_beta_params
            self.log(f"[Iteration {iteration}] HOMOGENEOUS: Sending global-DAG beta parameters for aggregation")
        
        pbytes = payload_size_bytes(participant_payload)
        self.log(f"[PAYLOAD] Client->Coordinator payload size: {pbytes} bytes")

        hist = self.load("payload_bytes_client_to_coord") or []
        hist.append(int(pbytes))
        self.store("payload_bytes_client_to_coord", hist)
        self.store("payload_bytes_client_to_coord_avg", float(sum(hist) / len(hist)))
        self.send_data_to_coordinator(participant_payload)

        if self.is_coordinator:
            self.log("LOCAL COMPUTATION to AGGREGATION")
            return AGGREGATION
        else:
            self.log("LOCAL COMPUTATION TO AWAIT AGGREGATION")
            return AWAIT_AGGREGATION
        
@app_state(AGGREGATION, Role.COORDINATOR)
class AggregationState(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION, Role.COORDINATOR)
        self.register_transition(FINAL, Role.COORDINATOR)

    def run(self):
        iteration = self.load('iteration')
        max_iterations = self.load('max_iterations')
        fl_min_iterations = self.load('fl_min_iterations')
        fl_patience = self.load('fl_patience')
        homogeneous = self.load('homogeneous')
        target = self.load('target')
        self.log(f"ITERATION : {iteration}")
        
        clients_payloads = self.gather_data()
        
        client_sizes = [payload_size_bytes(cp) for cp in clients_payloads]
        avg_client_size = float(sum(client_sizes) / len(client_sizes)) if client_sizes else 0.0

        self.log(f"[PAYLOAD] Avg Client->Coordinator payload size (this iter): {avg_client_size:.1f} bytes")
        self.log(f"[PAYLOAD] Client->Coordinator sizes: {client_sizes}")

        
        hist = self.load("payload_bytes_client_to_coord_avg_history") or []
        hist.append(avg_client_size)
        self.store("payload_bytes_client_to_coord_avg_history", hist)
        self.store("payload_bytes_client_to_coord_avg_overall", float(sum(hist) / len(hist)))
        
        clients_numbers = [cp['client_number'] for cp in clients_payloads]
        clients_pams_sparse = [cp['client_pam_sparse'] for cp in clients_payloads]
        clients_dataset_sizes = [cp['client_dataset_size'] for cp in clients_payloads]
        clients_bics = [cp['client_bic'] for cp in clients_payloads]
        clients_thresholds = [cp['client_threshold'] for cp in clients_payloads]
        clients_weights = [cds / sum(clients_dataset_sizes) for cds in clients_dataset_sizes]

        self.log(f"[COORDINATOR] WEIGHTS: {clients_weights}")
        
        
        coordinator = Coordinator()
        clients_pams = [coordinator.sparse_to_pam(sparse_pam) for sparse_pam in clients_pams_sparse]
        
        
        for i, sparse_pam in enumerate(clients_pams_sparse):
            num_nodes = len(sparse_pam['nodes'])
            num_edges = len(sparse_pam['edges'])
            total_possible = num_nodes * num_nodes
            sparsity = 100 * (1 - num_edges / total_possible) if total_possible > 0 else 0
            self.log(f"[COORDINATOR] Client {i+1}: {num_nodes} nodes, {num_edges} non-zero edges ({sparsity:.1f}% sparse)")
        
        
        avg_bic = sum(bic * weight for bic, weight in zip(clients_bics, clients_weights))
        self.log(f"[COORDINATOR] Iteration {iteration} Average BIC: {avg_bic:.2f}")
        
        
        global_threshold = sum(tau * weight for tau, weight in zip(clients_thresholds, clients_weights))
        self.log(f"[COORDINATOR] Iteration {iteration} Weighted average threshold: {global_threshold:.4f}")
        self.log(f"[COORDINATOR] Client thresholds: {[f'{tau:.2f}' for tau in clients_thresholds]}")
        
        
        avg_bic = sum(bic * weight for bic, weight in zip(clients_bics, clients_weights))
        self.log(f"[COORDINATOR] Iteration {iteration} Average BIC: {avg_bic:.2f}")
        
        if iteration == 1:
            self.store('bic_history', [avg_bic])
            self.store('best_avg_bic', avg_bic)
            self.store('iterations_without_improvement', 0)
            should_stop = False
        else:
            
            bic_history = self.load('bic_history')
            best_avg_bic = self.load('best_avg_bic')
            iterations_without_improvement = self.load('iterations_without_improvement')
            bic_history.append(avg_bic)
            self.store('bic_history', bic_history)
            
            if avg_bic > best_avg_bic:
                improvement = avg_bic - best_avg_bic
                self.log(f"[COORDINATOR] BIC improved by {improvement:.2f}!")
                self.store('best_avg_bic', avg_bic)
                self.store('iterations_without_improvement', 0)
                should_stop = False
            else:
                iterations_without_improvement += 1
                self.log(f"[COORDINATOR] No BIC improvement ({iterations_without_improvement}/{fl_patience})")
                self.store('iterations_without_improvement', iterations_without_improvement)
                
                
                if iteration >= fl_min_iterations and iterations_without_improvement >= fl_patience:
                    should_stop = True
                    self.log(f"[COORDINATOR] EARLY STOPPING TRIGGERED!")
                    self.log(f"[COORDINATOR] Completed {iteration} iterations (max: {max_iterations})")
                    self.log(f"[COORDINATOR] No improvement for {fl_patience} consecutive iterations")
                    self.log(f"[COORDINATOR] Best average BIC: {best_avg_bic:.2f}")
                    self.log(f"[COORDINATOR] Saved {max_iterations - iteration} iterations!")
                else:
                    should_stop = False
        
        coordinator = Coordinator()
        global_sum = np.sum(clients_numbers)
        global_pam = coordinator.aggregate_pams(clients_pams, clients_weights)
        self.store('global_sum', global_sum)
        self.store('global_sum_local', global_sum)
        self.store('global_pam', global_pam)
        self.store('global_pam_local', global_pam)
        
        self.log(f"[COORDINATOR] Creating global DAG structure for iteration {iteration}...")
        
        nodes = list(global_pam.index)
        edge_strengths = {(xi, xj): global_pam.loc[xi, xj]
                         for xi in nodes for xj in nodes
                         if xi != xj and global_pam.loc[xi, xj] > 0}
        
        if len(edge_strengths) == 0:
            self.log("[COORDINATOR] Warning: No edges in global PAM!")
            global_dag_edges = []
            global_dag_weights = {}
        else:
            self.log(f"[COORDINATOR] Applying global threshold: {global_threshold:.4f}")
            binary_pam = (global_pam >= global_threshold).astype(int)
            np.fill_diagonal(binary_pam.values, 0)
            global_dag = coordinator.create_dag(binary_pam, edge_strengths, verbose=False)
            isolated_nodes = [node for node in global_dag.nodes() if global_dag.degree(node) == 0]
            global_dag.remove_nodes_from(isolated_nodes)
            
            if len(isolated_nodes) > 0:
                self.log(f"[COORDINATOR] Removed {len(isolated_nodes)} isolated nodes")
            
            self.log(f"[COORDINATOR] Global DAG: {global_dag.number_of_nodes()} nodes, {global_dag.number_of_edges()} edges")
            
            global_dag_edges = list(global_dag.edges())
            global_dag_weights = {(u, v): float(global_pam.loc[u, v]) 
                                 for (u, v) in global_dag_edges
                                 if u in global_pam.index and v in global_pam.columns}
        
        global_pam_sparse = coordinator.pam_to_sparse(global_pam)
        total_possible = len(global_pam) * len(global_pam)
        num_nonzero = len(global_pam_sparse['edges'])
        global_sparsity = 100 * (1 - num_nonzero / total_possible) if total_possible > 0 else 0
        self.log(f"[COORDINATOR] Global PAM: {len(global_pam)} nodes, {num_nonzero} non-zero edges ({global_sparsity:.1f}% sparse)")
        self.store('global_dag_edges_local', global_dag_edges)
        self.store('global_threshold_local', global_threshold)
        self.log(f"[COORDINATOR] Stored global DAG edges for next iteration: {len(global_dag_edges)} edges")

        global_betas = None
        if homogeneous and iteration > 1:  
            try:
                clients_betas = [cp.get('client_betas') for cp in clients_payloads]
                clients_betas_valid = [cb for cb in clients_betas if cb is not None]
                
                if len(clients_betas_valid) > 0:
                    valid_indices = [i for i, cb in enumerate(clients_betas) if cb is not None]
                    valid_weights = [clients_weights[i] for i in valid_indices]
    
                    self.log(f"[COORDINATOR] HOMOGENEOUS MODE: Aggregating beta parameters from {len(clients_betas_valid)} clients")
                    global_betas = coordinator.aggregate_betas(clients_betas_valid, valid_weights)
                    self.log(f"[COORDINATOR] Aggregated betas for {len(global_betas)} nodes")
                else:
                    self.log(f"[COORDINATOR] No beta parameters received from clients")
            except Exception as e:
                self.log(f"[COORDINATOR] ERROR aggregating betas: {e}")
                self.log(traceback.format_exc())

        coordinator_payload = {
            "global_sum": global_sum,
            "global_pam_sparse": global_pam_sparse,  
            "global_dag_edges": global_dag_edges,
            "global_dag_weights": global_dag_weights,
            "global_threshold": global_threshold,
            "should_stop": should_stop,
            "homogeneous": homogeneous,
            "Message": f"Iteration {iteration} complete"
        }
        
        if global_betas is not None:
            coordinator_payload["global_betas"] = global_betas
            self.log(f"[COORDINATOR] Broadcasting global beta parameters")
        
        
        cbytes = payload_size_bytes(coordinator_payload)
        self.log(f"[PAYLOAD] Coordinator->Clients payload size: {cbytes} bytes")

        hist = self.load("payload_bytes_coord_to_client_history") or []
        hist.append(int(cbytes))
        self.store("payload_bytes_coord_to_client_history", hist)
        self.store("payload_bytes_coord_to_client_avg", float(sum(hist) / len(hist)))
        
        if len(global_dag_edges) > 0:
            try:
                output_dir = self.load('output_dir')
                dag_viz = nx.DiGraph()
                dag_viz.add_edges_from(global_dag_edges)
    
                viz_filename = os.path.join(output_dir, f"coordinator_global_dag_iteration_{iteration}.png")
                coordinator.visualize_network(
                    dag_viz,
                    viz_filename,
                    target=target,
                    edge_weights=global_dag_weights
                )
                self.log(f"[COORDINATOR] Global DAG visualization saved to: {viz_filename}")
            except Exception as e:
                self.log(f"[COORDINATOR] Warning: Could not save DAG visualization: {e}")
        
        self.broadcast_data(coordinator_payload)

        iteration += 1
        self.store('iteration', iteration)
        
        
        if should_stop:
            self.log("[COORDINATOR] Early stopping - transitioning to FINAL")
            return FINAL
        elif iteration <= max_iterations:
            return LOCAL_COMPUTATION
        else:
            self.log("[COORDINATOR] Max iterations reached - transitioning to FINAL")
            return FINAL

@app_state(AWAIT_AGGREGATION, Role.PARTICIPANT)
class AwaitAggregationState(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION, Role.PARTICIPANT)
        self.register_transition(FINAL, Role.PARTICIPANT)

    def run(self):
        iteration = self.load('iteration')
        max_iterations = self.load('max_iterations')
        target = self.load('target')
        self.log(f"ITERATION : {iteration}")

        coordinator_payload = self.await_data()
        cbytes = payload_size_bytes(coordinator_payload)
        self.log(f"[PAYLOAD] Coordinator->Client payload size: {cbytes} bytes")

        hist = self.load("payload_bytes_coord_to_client") or []
        hist.append(int(cbytes))
        self.store("payload_bytes_coord_to_client", hist)
        self.store("payload_bytes_coord_to_client_avg", float(sum(hist) / len(hist)))
        
        global_sum = coordinator_payload['global_sum']
        global_pam_sparse = coordinator_payload['global_pam_sparse']
        global_dag_edges = coordinator_payload.get('global_dag_edges', [])
        global_dag_weights = coordinator_payload.get('global_dag_weights', {})
        global_threshold = coordinator_payload.get('global_threshold', 0.0)
        should_stop = coordinator_payload.get('should_stop', False)
        homogeneous = coordinator_payload.get('homogeneous', False)
        global_betas = coordinator_payload.get('global_betas', None)
        
        participant = Client()
        global_pam = participant.sparse_to_pam(global_pam_sparse)
        
        self.store('global_sum_local', global_sum)
        self.store('global_pam_local', global_pam)
        self.store('global_dag_edges_local', global_dag_edges)  
        self.store('global_threshold_local', global_threshold)  
        
        if homogeneous and global_betas is not None:
            self.store('global_betas_local', global_betas)
            self.log(f"[CLIENT] Received global beta parameters for {len(global_betas)} nodes")
        
        
        if len(global_dag_edges) > 0:
            global_dag = nx.DiGraph()
            global_dag.add_edges_from(global_dag_edges)
            
            self.log(f"[CLIENT] Received global DAG: {global_dag.number_of_nodes()} nodes, {global_dag.number_of_edges()} edges (threshold: {global_threshold:.4f})")
            output_dir = self.load('output_dir')
            participant = Client()
            participant.visualize_network(
                global_dag, 
                os.path.join(output_dir, f"global_dag_iteration_{iteration}.png"),
                target=target,
                edge_weights=global_dag_weights
            )
            self.log(f"[CLIENT] Global DAG at Iteration {iteration}: {global_dag.edges()}")
            self.log(f"[CLIENT] Global DAG visualization saved for iteration {iteration}")
        else:
            self.log(f"[CLIENT] No global DAG edges for iteration {iteration}")
        
        iteration += 1
        self.store('iteration', iteration)
        
        if should_stop:
            self.log(f"[CLIENT] Early stopping signal received from coordinator")
            return FINAL
        elif iteration <= max_iterations:
            return LOCAL_COMPUTATION
        else:
            return FINAL
        

@app_state(FINAL, Role.BOTH)
class FinalState(AppState):
    def register(self):
        self.register_transition(TERMINAL, Role.BOTH)

    def run(self):
        self.log("FINAL STATE - Creating final visualizations and summary")
        
        output_dir = self.load('output_dir')
        target = self.load('target')
        participant = Client()
        
        try:
            metrics_history = self.load('metrics_history')
            
            participant.visualize_metrics_over_iterations(
                metrics_history,
                os.path.join(output_dir, 'metrics_evolution.png')
            )
            
            self.log("="*60)
            self.log("FINAL RESULTS SUMMARY")
            self.log("="*60)
            self.log(f"Total Iterations: {len(metrics_history['iteration'])}")
            self.log(f"Final BIC: {metrics_history['bic'][-1]:.2f}")
            self.log(f"Final Accuracy: {metrics_history['accuracy_mean'][-1]:.4f} ± {metrics_history['accuracy_std'][-1]:.4f}")
            self.log(f"Final F1-Score: {metrics_history['f1_mean'][-1]:.4f} ± {metrics_history['f1_std'][-1]:.4f}")
            self.log(f"Final AUROC: {metrics_history['auroc_mean'][-1]:.4f} ± {metrics_history['auroc_std'][-1]:.4f}")
            self.log(f"Final Edges: {metrics_history['edges'][-1]}")
            self.log(f"Final Threshold: {metrics_history['threshold'][-1]:.2f}")
            self.log("="*60)
        except Exception as e:
            self.log(f"Warning: Could not create final visualization: {e}")
        
        try:
            self.log("Generating predictions on full dataset...")
            input_dir = self.load('input_dir')
            dataset_loc = self.load('dataset_loc')
            dataset_path = os.path.join(input_dir, dataset_loc)
            
            if os.path.exists(dataset_path):
                dataset = pd.read_csv(dataset_path)
                final_dag = self.load('final_dag')
                if final_dag is not None:
                    predictions_path = os.path.join(output_dir, 'predictions_with_probabilities.csv')
                    self.log("Re-estimating parameters with class balancing for predictions...")
                    participant = Client()
                    final_params_balanced = participant.estimate_multilogit_params(
                        final_dag,
                        dataset
                    )
                    
                    df_with_predictions = participant.save_predictions_to_csv(
                        dag=final_dag,
                        params=final_params_balanced,  
                        dataset=dataset,
                        output_path=predictions_path,
                        target=target
                    )
                    
                    self.log(f"  Predictions saved to: {predictions_path}")
                    self.log(f"  Dataset shape: {df_with_predictions.shape}")
                    self.log(f"  Columns added:")
                    self.log(f"    - Predicted_{target}")
                    self.log(f"    - Probability_<class> for each class")
                    self.log(f"    - Prediction_Confidence")
                    
                    pred_counts = df_with_predictions[f"Predicted_{target}"].value_counts()
                    self.log(f"  Prediction distribution:")
                    for label, count in pred_counts.items():
                        pct = count / len(df_with_predictions) * 100
                        self.log(f"    {label}: {count} ({pct:.1f}%)")

                    y_true = df_with_predictions[target].values
                    y_pred = df_with_predictions[f"Predicted_{target}"].values
                    
                    classes = sorted(dataset[target].unique())
                    n_classes = len(classes)
                    final_accuracy = accuracy_score(y_true, y_pred)
                    final_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    final_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    
                    try:
                        if n_classes > 2:
                            
                            prob_cols = [f'Probability_{cls}' for cls in classes]
                            y_proba = df_with_predictions[prob_cols].values
                            final_auroc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                        else:
                            prob_col = f'Probability_{classes[1]}'
                            y_proba = df_with_predictions[prob_col].values
                            final_auroc = roc_auc_score(y_true, y_proba)
                    except Exception as e:
                        final_auroc = None
                        self.log(f"  Could not compute AUROC: {e}")
                    
                    self.log("")
                    self.log("="*60)
                    self.log("FINAL PREDICTIONS METRICS")
                    self.log("="*60)
                    self.log(f"  Accuracy:        {final_accuracy:.4f}")
                    self.log(f"  F1-Score (Macro):    {final_f1_macro:.4f}")
                    self.log(f"  F1-Score (Weighted): {final_f1_weighted:.4f}")
                    if final_auroc is not None:
                        self.log(f"  AUROC:           {final_auroc:.4f}")
                    self.log("="*60)
                else:
                    self.log("Warning: No final DAG found, skipping predictions")
            else:
                self.log(f"Warning: Dataset file not found at {dataset_path}")
        except Exception as e:
            self.log(f"Error generating predictions: {e}")
            self.log(traceback.format_exc())
    
        start_time = self.load('start_time')
        if start_time is not None:
            total_time_seconds = time.time() - start_time
            total_time_minutes = total_time_seconds / 60.0
            
            self.log("")
            self.log("="*60)
            self.log("COMMUNICATION AND TIME STATISTICS")
            self.log("="*60)
            self.log(f"Total Time: {total_time_minutes:.2f} minutes ({total_time_seconds:.2f} seconds)")
            self.log("")
            
            
            if self.is_coordinator:
                
                client_avg = self.load("payload_bytes_client_to_coord_avg_overall")
                if client_avg is not None:
                    self.log(f"Average Client->Coordinator Payload Size: {client_avg / 1024:.2f} kB")
                coord_avg = self.load("payload_bytes_coord_to_client_avg")
                if coord_avg is not None:
                    self.log(f"Average Coordinator->Clients Payload Size: {coord_avg / 1024:.2f} kB")
            else:
                client_avg = self.load("payload_bytes_client_to_coord_avg")
                if client_avg is not None:
                    self.log(f"Average Client->Coordinator Payload Size: {client_avg / 1024:.2f} kB")
                coord_avg = self.load("payload_bytes_coord_to_client_avg")
                if coord_avg is not None:
                    self.log(f"Average Coordinator->Client Payload Size: {coord_avg / 1024:.2f} kB")

            self.log("="*60)
            self.log("")
        
        self.log("FINAL to TERMINAL")
        return TERMINAL