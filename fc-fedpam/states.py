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
import itertools

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='pgmpy')
warnings.filterwarnings('ignore', message='.*Replacing existing CPD.*')
warnings.filterwarnings('ignore', message='.*pgmpy.*')

from algorithms import Client, Coordinator

INITIAL = 'initial'
READ_INPUT = 'read input'
SEND_DATASET_SIZE = 'send dataset size'  
AGGREGATE_DATASET_SIZE = 'aggregate dataset size'  
AWAIT_WEIGHTS = 'await weights'  
LOCAL_COMPUTATION_1 = 'local computation 1'
SEND_THRESHOLD = 'send threshold'
AGGREGATE_THRESHOLDS = 'aggregate thresholds'
AWAIT_THRESHOLD = 'await thresholds'
LOCAL_COMPUTATION_2 = 'local computation 2'
AGGREGATION = 'aggregation'
AWAIT_AGGREGATION = 'await aggregation'
SEND_BETAS = 'send betas' #smpc
AGGREGATE_BETAS = 'aggregate betas'
AWAIT_AGGREGATED_BETAS = 'await aggregated betas'
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

def _sorted_unique_values(series: pd.Series):
    vals = series.dropna().unique().tolist()
    try:
        return sorted(vals)
    except Exception:
        return sorted(vals, key=lambda x: str(x))

def _normalize_prob_map(classes, probs, target_states):
    if probs is None:
        base = 1.0 / max(len(target_states), 1)
        return {str(s): float(base) for s in target_states}
    
    arr = np.asarray(probs, dtype=float).reshape(-1)
    if arr.size == 1 and len(target_states) == 2:
        arr = np.array([1.0 - arr[0], arr[0]], dtype=float)
    
    if classes is None:
        if arr.size < len(target_states):
            arr = np.pad(arr, (0, len(target_states) - arr.size), mode="constant", constant_values=0.0)
        elif arr.size > len(target_states):
            arr = arr[:len(target_states)]
        total = float(arr.sum())
        if total <= 0:
            arr = np.ones(len(target_states), dtype=float) / max(len(target_states), 1)
        else:
            arr = arr / total
        return {str(target_states[i]): float(arr[i]) for i in range(len(target_states))}
    
    class_list = [str(c) for c in np.asarray(classes).tolist()]
    if arr.size < len(class_list):
        arr = np.pad(arr, (0, len(class_list) - arr.size), mode="constant", constant_values=0.0)
    elif arr.size > len(class_list):
        arr = arr[:len(class_list)]
    
    tmp = {str(class_list[i]): float(arr[i]) for i in range(len(class_list))}
    out = {str(s): float(tmp.get(str(s), 0.0)) for s in target_states}
    total = float(sum(out.values()))
    if total <= 0:
        base = 1.0 / max(len(target_states), 1)
        return {str(s): float(base) for s in target_states}
    return {k: float(v / total) for k, v in out.items()}

def _empirical_conditional_probs(dataset, node, states, parents, parent_vals, laplace=1.0):
    if len(parents) == 0:
        counts = dataset[node].value_counts().to_dict()
    else:
        mask = np.ones(len(dataset), dtype=bool)
        for p, v in zip(parents, parent_vals):
            mask &= (dataset[p] == v).values
        subset = dataset.loc[mask, node]
        counts = subset.value_counts().to_dict()
    
    probs = {}
    total = 0.0
    for s in states:
        c = float(counts.get(s, 0.0) + laplace)
        probs[str(s)] = c
        total += c
    if total <= 0:
        base = 1.0 / max(len(states), 1)
        return {str(s): float(base) for s in states}
    return {k: float(v / total) for k, v in probs.items()}

def _build_cpts_from_params(dag_edges, params, dataset, participant):
    params = params or {}
    G = nx.DiGraph()
    G.add_edges_from(dag_edges or [])
    if dataset is not None:
        G.add_nodes_from(dataset.columns.tolist())
    for nid in params.keys():
        G.add_node(nid)
    
    cpts = {}
    node_order = list(nx.topological_sort(G)) if nx.is_directed_acyclic_graph(G) else list(G.nodes())
    for node in node_order:
        parents = list(G.predecessors(node))
        node_states = None
        node_params = params.get(node)
        if isinstance(node_params, dict) and node_params.get("classes") is not None:
            node_states = [str(c) for c in np.asarray(node_params.get("classes")).tolist()]
        elif dataset is not None and node in dataset.columns:
            node_states = [str(v) for v in _sorted_unique_values(dataset[node])]
        else:
            node_states = ["0", "1"]
        
        parent_states = {}
        for p in parents:
            if dataset is not None and p in dataset.columns:
                parent_states[p] = [str(v) for v in _sorted_unique_values(dataset[p])]
            elif isinstance(params.get(p), dict) and params[p].get("classes") is not None:
                parent_states[p] = [str(c) for c in np.asarray(params[p].get("classes")).tolist()]
            else:
                parent_states[p] = ["0", "1"]
        
        cpt_rows = {}
        if len(parents) == 0:
            if isinstance(node_params, dict):
                probs = _normalize_prob_map(node_params.get("classes"), node_params.get("intercept"), node_states)
            elif dataset is not None and node in dataset.columns:
                probs = _empirical_conditional_probs(dataset, node, node_states, [], [])
            else:
                base = 1.0 / max(len(node_states), 1)
                probs = {str(s): float(base) for s in node_states}
            cpt_rows["__ROOT__"] = probs
        else:
            parent_lists = [parent_states[p] for p in parents]
            for combo in itertools.product(*parent_lists):
                combo_key = "|".join([f"{parents[i]}={combo[i]}" for i in range(len(parents))])
                probs = None
                if isinstance(node_params, dict):
                    try:
                        parent_dict = {}
                        for i in range(len(parents)):
                            parent_name = parents[i]
                            parent_val = combo[i]
                            
                            try:
                                parent_dict[parent_name] = float(parent_val)
                            except (ValueError, TypeError):
                                parent_dict[parent_name] = parent_val
                        
                        parent_df = pd.DataFrame([parent_dict])
                        pred = participant._predict_node_proba(parent_df, node_params)
                        probs = _normalize_prob_map(node_params.get("classes"), pred[0], node_states)
                    except Exception as e:
                        import traceback
                        print(f"[DEBUG] Prediction failed for {node} with parents {combo}: {e}")
                        traceback.print_exc()
                        probs = None
                
                if probs is None:
                    if dataset is not None and node in dataset.columns and all(p in dataset.columns for p in parents):
                        casted_combo = []
                        for i, p in enumerate(parents):
                            target_dtype = dataset[p].dtype
                            raw_val = combo[i]
                            try:
                                casted = target_dtype.type(raw_val)
                            except Exception:
                                casted = raw_val
                            casted_combo.append(casted)
                        probs = _empirical_conditional_probs(dataset, node, node_states, parents, casted_combo)
                    else:
                        base = 1.0 / max(len(node_states), 1)
                        probs = {str(s): float(base) for s in node_states}
                
                cpt_rows[combo_key] = probs
        
        cpts[str(node)] = {
            "parents": [str(p) for p in parents],
            "states": [str(s) for s in node_states],
            "cpt": cpt_rows
        }
    
    return cpts

def _build_network_export_payload(dag_edges, params, dataset, participant):
    dag_edges = dag_edges or []
    params = params or {}
    
    node_ids = set()
    for e in dag_edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            node_ids.add(str(e[0]))
            node_ids.add(str(e[1]))
    if dataset is not None:
        for col in dataset.columns:
            node_ids.add(str(col))
    for nid in params.keys():
        node_ids.add(str(nid))
    
    nodes = []
    for nid in sorted(node_ids):
        if dataset is not None and nid in dataset.columns:
            states = [str(v) for v in _sorted_unique_values(dataset[nid])]
        elif isinstance(params.get(nid), dict) and params[nid].get("classes") is not None:
            states = [str(c) for c in np.asarray(params[nid].get("classes")).tolist()]
        else:
            states = ["0", "1"]
        nodes.append({
            "id": nid,
            "label": nid,
            "description": "",
            "states": states
        })
    
    edges_obj = []
    for e in dag_edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            edges_obj.append({"from": str(e[0]), "to": str(e[1])})
    
    cpts = _build_cpts_from_params(dag_edges, params, dataset, participant)
    return {
        "nodes": nodes,
        "edges": edges_obj,
        "cpts": cpts
    }

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
        self.register_transition(SEND_DATASET_SIZE, Role.BOTH) 
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
            config_file = yaml.safe_load(cfp, Loader = yaml.FullLoader)

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
            self.store('dataset_size', len(splits[client_split_path])) 

            client = roles[client_split_path]
            self.store('client_instance', client)
            self.store('iteration', 1)
            
            self.store('metrics_history', {
                'iteration': [],
                'bic': [],
                'accuracy_mean': [],
                'accuracy_std': [],
                'f1_mean': [],
                'f1_std': [],
                'auroc_mean': [],
                'auroc_std': [],
                'edges': [],
                'threshold': []
            })

            self.log("READ INPUT TO SEND_DATASET_SIZE")
            return SEND_DATASET_SIZE

@app_state(SEND_DATASET_SIZE, Role.BOTH)
class SendDatasetSizeState(AppState):
    def register(self):
        self.register_transition(AGGREGATE_DATASET_SIZE, Role.COORDINATOR)
        self.register_transition(AWAIT_WEIGHTS, Role.PARTICIPANT)
    
    def run(self):
        dataset_size = self.load('dataset_size')
        self.log(f"[PARTICIPANT] Local dataset size: {dataset_size}")
        
        self.send_data_to_coordinator(dataset_size)
        
        if self.is_coordinator:
            self.log("[COORDINATOR] Sent dataset size -> AGGREGATE_DATASET_SIZE")
            return AGGREGATE_DATASET_SIZE
        else:
            self.log("[PARTICIPANT] Sent dataset size -> AWAIT_WEIGHTS")
            return AWAIT_WEIGHTS


@app_state(AGGREGATE_DATASET_SIZE, Role.COORDINATOR)
class AggregateDatasetSizeState(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION_1, Role.COORDINATOR)
    
    def run(self):
        client_data_sizes = self.gather_data()
        total_size = sum(client_data_sizes)
        num_clients = len(client_data_sizes)        
        
        self.log(f"[COORDINATOR] Total dataset size: {total_size}")
        self.log(f"[COORDINATOR] Number of clients: {num_clients}")
        self.store('total_data_size', total_size)
        client_prop = 1/num_clients
        weight_payload = {
            'total_data_size': total_size,
            'client_prop': client_prop
        }
        self.broadcast_data(weight_payload, send_to_self=True)
        
        self.log("[COORDINATOR] Broadcasted total size -> LOCAL_COMPUTATION_1")
        return LOCAL_COMPUTATION_1


@app_state(AWAIT_WEIGHTS, Role.PARTICIPANT)
class AwaitWeightsState(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION_1, Role.PARTICIPANT)
    
    def run(self):
        weight_payload = self.await_data()
        total_data_size = weight_payload['total_data_size']
        client_prop = weight_payload['client_prop']
        self.store('total_data_size', total_data_size)
        self.store('client_prop', client_prop)

        self.log(f"[PARTICIPANT] Received total size: {total_data_size}")
        return LOCAL_COMPUTATION_1
    
        
@app_state(LOCAL_COMPUTATION_1, Role.BOTH)
class LocalComputation1(AppState):
    def register(self):
        self.register_transition(AGGREGATE_THRESHOLDS, Role.COORDINATOR)
        self.register_transition(AWAIT_THRESHOLD, Role.PARTICIPANT)
    
    def run(self):
        output_dir = self.load('output_dir')
        iteration = self.load('iteration')
        dataset = self.load('dataset')
        dataset_size = self.load('dataset_size')
        
        target = self.load('target')
        mu = self.load('mu')
        self.log(f"ITERATION : {iteration}")

        participant = Client()
        
        total_dataset_size = self.load('total_data_size')
        local_weight = dataset_size / total_dataset_size
        self.store('local_weight', local_weight)

        if iteration == 1:
            self.store('metrics_history', {
                'iteration': [],
                'bic': [],
                'accuracy_mean': [],
                'accuracy_std': [],
                'f1_mean': [],
                'f1_std': [],
                'auroc_mean': [],
                'auroc_std': [],
                'edges': [],
                'threshold': []
            })

            self.log("="*60)
            self.log(f"Iteration {iteration}: Initializing local PAM via bootstrap")
            self.log("="*60)

            # PAM Computation
            bootstrap_iterations = self.load('bootstrap_iterations')
            bootstrap_min_iterations = self.load('bootstrap_min_iterations')
            bootstrap_patience = self.load('bootstrap_patience')
            bootstrap_output = participant.create_prob_adj_matrix(
                dataset = dataset,
                target = target,
                num_iterations = bootstrap_iterations,
                min_iterations = bootstrap_min_iterations,
                patience = bootstrap_patience 
            )

            if isinstance(bootstrap_output, tuple):
                local_pam = bootstrap_output[0]
            else:
                local_pam = bootstrap_output

            self.log(f"[Iteration {iteration}] Local PAM shape: {local_pam.shape}")
            participant.visualize_pam(local_pam, os.path.join(output_dir, f"local_pam_{iteration}.png"))
            self.store('local_pam', local_pam)
            
            self.log("Local CMI Computation")
            local_cmi = participant.compute_cmi_reward_matrix(dataset, local_pam)
            self.store('local_cmi', local_cmi)

        else:
            self.log("="*60)
            self.log(f"Iteration {iteration}: Optimizing local PAM")
            self.log("="*60)

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

        self.log(f"[Iteration {iteration}] Running BIC-based threshold search...")
        best_dag, best_tau, best_bic, best_params = participant.bic_threshold_search(
            local_pam, dataset
        )

        if best_dag is None:
            self.log(f"[Iteration {iteration}] WARNING: No valid DAG found! Using empty DAG.")
            best_dag = nx.DiGraph()
            best_dag.add_nodes_from(dataset.columns.tolist())
            best_params = {}
            best_tau = 0.0
            best_bic = -float('inf')

        self.store('best_dag', best_dag)
        self.store('best_tau', best_tau)
        self.store('best_bic', best_bic)
        self.store('best_params', best_params)

        self.send_data_to_coordinator(best_tau*local_weight, send_to_self=True, use_smpc=True)
        if self.is_coordinator:
            return AGGREGATE_THRESHOLDS
        else:
            return AWAIT_THRESHOLD
        
@app_state(AGGREGATE_THRESHOLDS, Role.COORDINATOR)
class AggregateThresholds(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION_2, Role.COORDINATOR)

    def run(self):
        tau_avg = self.gather_data(use_smpc=True)
        if isinstance(tau_avg, list) and len(tau_avg) > 0:
            tau_avg = tau_avg[0]
        self.log(f"[COORDINATOR] Average Tau is {tau_avg}")
        self.store('tau_avg', tau_avg)
        message = "Thresholds aggregated"
        self.broadcast_data(message)
        return LOCAL_COMPUTATION_2
    
@app_state(AWAIT_THRESHOLD, Role.PARTICIPANT)
class AwaitThresholdMessage(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION_2, Role.PARTICIPANT)

    def run(self):
        message = self.await_data()
        return LOCAL_COMPUTATION_2
    
@app_state(LOCAL_COMPUTATION_2, Role.BOTH)
class LocalComputation2(AppState):
    def register(self):
        self.register_transition(AGGREGATION, Role.COORDINATOR)
        self.register_transition(AWAIT_AGGREGATION, Role.PARTICIPANT)

    def run(self):
        output_dir = self.load('output_dir')
        iteration = self.load('iteration')
        homogeneous = self.load('homogeneous')
        dataset = self.load('dataset')
        dataset_size = self.load('dataset_size')
        target = self.load('target')
        local_dag = self.load('best_dag')
        local_tau = self.load('best_tau')
        local_params = self.load('best_params')
        local_bic = self.load('best_bic')
        local_pam = self.load('local_pam')
        global_beta_params = None

        participant = Client()

        if homogeneous == True:
            if iteration == 1:
                self.log(f"[Iteration {iteration}] HOMOGENEOUS MODE (initial): Using local DAG for parameter estimation")
                global_dag = local_dag
            else:
                self.log(f"[Iteration {iteration}] HOMOGENEOUS MODE: Using global DAG for parameter estimation")

                global_dag_edges = self.load('global_dag_edges_local')
                if not global_dag_edges or len(global_dag_edges) == 0:
                    raise RuntimeError(f"[Iteration {iteration}] HOMOGENEOUS MODE: No global DAG available!")
                global_dag = nx.DiGraph()
                global_dag.add_edges_from(global_dag_edges)
                self.log(f"[Iteration {iteration}] Global DAG: "
                         f"{global_dag.number_of_nodes()} nodes, {global_dag.number_of_edges()} edges")
                
                global_beta_params = participant.estimate_multilogit_params(global_dag, dataset)

        else:
            self.log(f"[Iteration {iteration}] HETEROGENEOUS MODE: Using refined local PAM and local DAG for personalized parameters")

            self.log(f"[Iteration {iteration}] BEST TAU: {local_tau}")

        self.log(f"[Iteration {iteration}] BEST BIC: {local_bic}")
        self.log(f"[Iteration {iteration}] EDGES: {local_dag.number_of_edges()}")
        self.log(f"[Iteration {iteration}] NUM PARAMS: {participant._count_multilogit_params(local_dag, dataset)}")

        self.log(f"[Iteration {iteration}] Evaluating predictive performance...")
        cv_results = participant.evaluate_kfold_cv(
            local_dag, dataset, target=target, k=5, random_state=23
        )
        self.log(f"[Iteration {iteration}] Accuracy: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        self.log(f"[Iteration {iteration}] F1-Score: {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        self.log(f"[Iteration {iteration}] AUROC-OVR: {cv_results['auroc_mean']:.4f} ± {cv_results['auroc_std']:.4f}")
        
        metrics_history = self.load('metrics_history')
        metrics_history['iteration'].append(iteration)
        metrics_history['bic'].append(local_bic)
        metrics_history['accuracy_mean'].append(cv_results['accuracy_mean'])
        metrics_history['accuracy_std'].append(cv_results['accuracy_std'])
        metrics_history['f1_mean'].append(cv_results['f1_mean'])  
        metrics_history['f1_std'].append(cv_results['f1_std'])    
        metrics_history['auroc_mean'].append(cv_results['auroc_mean'])
        metrics_history['auroc_std'].append(cv_results['auroc_std'])
        metrics_history['edges'].append(local_dag.number_of_edges())
        metrics_history['threshold'].append(local_tau)
        self.store('metrics_history', metrics_history)
        self.store('final_dag', local_dag)

        edge_weights = {}
        for (u, v) in local_dag.edges():
            if u in local_pam.index and v in local_pam.columns:
                edge_weights[(u, v)] = float(local_pam.loc[u, v])

        participant.visualize_network(
            local_dag,
            os.path.join(output_dir, f"local_dag_{iteration}.png"),
            target=target,
            edge_weights=edge_weights
        )

        if homogeneous:
            self.log(f"[CLIENT] Homogeneous mode - Local DAG at Iteration {iteration} "
                     f"(weighted by refined local PAM): {local_dag.edges()}")
        else:
            self.log(f"[CLIENT] Heterogeneous mode - Local DAG at Iteration {iteration} "
                     f"(weighted by refined local PAM): {local_dag.edges()}")

        participant.visualize_pam(local_pam, os.path.join(output_dir, f"refined_local_pam_{iteration}.png"))
        if iteration > 1:
            global_pam_viz = self.load('global_pam_local')
            participant.visualize_pam(global_pam_viz, os.path.join(output_dir, f"global_pam_{iteration}.png"))
        
        participant_sparse_pam = participant.pam_to_sparse(local_pam)
        participant_payload = {
            "client_pam_sparse": participant_sparse_pam,
            "client_dataset_size": dataset_size,
            "client_bic": local_bic
        }

        # SMPC-based Parameter Aggregation
        if homogeneous and iteration > 1 and global_beta_params is not None:
            self.store('client_betas_to_send', global_beta_params)
            self.log(f"[Iteration {iteration}] HOMOGENEOUS: Stored beta parameters for SMPC aggregation")
        else:
            self.store('client_betas_to_send', None)

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
        self.register_transition(SEND_BETAS, Role.COORDINATOR)
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
        
        clients_pams_sparse = [cp['client_pam_sparse'] for cp in clients_payloads]
        clients_dataset_sizes = [cp['client_dataset_size'] for cp in clients_payloads]
        clients_bics = [cp['client_bic'] for cp in clients_payloads]
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
        
        
        global_threshold = self.load('tau_avg')
        
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
        global_pam = coordinator.aggregate_pams(clients_pams, clients_weights)
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

        if iteration == 1 and homogeneous:
            dataset = self.load('dataset')  
            node_order, param_positions, total_params, beta_metadata = coordinator.create_beta_ordering_with_metadata(
                global_dag_edges, 
                dataset
            )
            self.store('node_order', node_order)
            self.store('param_positions', param_positions)
            self.store('total_params', total_params)
            self.store('beta_metadata', beta_metadata)
            self.log(f"[COORDINATOR] Created beta parameter ordering: {len(node_order)} nodes, {total_params} total parameters")
            self.log(f"[COORDINATOR] Stored beta metadata for {len(beta_metadata)} nodes")
        
        self.store('clients_dataset_sizes', clients_dataset_sizes)
        self.store('clients_weights', clients_weights)
        self.store('should_stop', should_stop)
        
        coordinator_payload = {
            "global_pam_sparse": global_pam_sparse,  
            "global_dag_edges": global_dag_edges,
            "global_dag_weights": global_dag_weights,
            "global_threshold": global_threshold,
            "should_stop": should_stop,
            "homogeneous": homogeneous,
            "Message": f"Iteration {iteration} complete - metadata"
        }
        
        if iteration == 1 and homogeneous:
            coordinator_payload["node_order"] = self.load('node_order')
            coordinator_payload["param_positions"] = self.load('param_positions')
            coordinator_payload["total_params"] = self.load('total_params')
            coordinator_payload["beta_metadata"] = self.load('beta_metadata')
        
        cbytes = payload_size_bytes(coordinator_payload)
        self.log(f"[PAYLOAD] Coordinator->Clients metadata payload size: {cbytes} bytes")

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
        self.log("[COORDINATOR] Transitioning to SEND_BETAS for SMPC aggregation")
        return SEND_BETAS

@app_state(AWAIT_AGGREGATION, Role.PARTICIPANT)
class AwaitAggregationState(AppState):
    def register(self):
        self.register_transition(SEND_BETAS, Role.PARTICIPANT)
        self.register_transition(FINAL, Role.PARTICIPANT)

    def run(self):
        iteration = self.load('iteration')
        max_iterations = self.load('max_iterations')
        target = self.load('target')
        self.log(f"ITERATION : {iteration}")

        coordinator_payload = self.await_data()
        cbytes = payload_size_bytes(coordinator_payload)
        self.log(f"[PAYLOAD] Coordinator->Client metadata payload size: {cbytes} bytes")

        hist = self.load("payload_bytes_coord_to_client_history") or []
        hist.append(int(cbytes))
        self.store("payload_bytes_coord_to_client_history", hist)
        self.store("payload_bytes_coord_to_client_avg", float(sum(hist) / len(hist)))

        global_sum = coordinator_payload.get('global_sum')
        global_pam_sparse = coordinator_payload.get('global_pam_sparse')
        global_dag_edges = coordinator_payload.get('global_dag_edges')
        global_dag_weights = coordinator_payload.get('global_dag_weights')
        global_threshold = coordinator_payload.get('global_threshold')
        homogeneous = coordinator_payload.get('homogeneous')
        
        if iteration == 1 and homogeneous:
            node_order = coordinator_payload.get('node_order')
            param_positions = coordinator_payload.get('param_positions')
            total_params = coordinator_payload.get('total_params')
            beta_metadata = coordinator_payload.get('beta_metadata')
            
            if node_order is not None:
                self.store('node_order', node_order)
                self.store('param_positions', param_positions)
                self.store('total_params', total_params)
                self.store('beta_metadata', beta_metadata)
                self.log(f"[PARTICIPANT] Received beta parameter ordering: {len(node_order)} nodes, {total_params} total parameters")
                if beta_metadata:
                    self.log(f"[PARTICIPANT] Stored beta metadata for {len(beta_metadata)} nodes")

        self.store('global_sum_local', global_sum)
        self.store('global_threshold_local', global_threshold)
        self.store('global_dag_edges_local', global_dag_edges)

        coordinator = Coordinator()
        global_pam = coordinator.sparse_to_pam(global_pam_sparse)
        self.store('global_pam_local', global_pam)

        if global_dag_edges is not None and len(global_dag_edges) > 0:
            self.log(f"[PARTICIPANT] Received global DAG: {len(global_dag_edges)} edges")
            self.log(f"[PARTICIPANT] Edges: {global_dag_edges[:10]}..." if len(global_dag_edges) > 10 else f"[PARTICIPANT] Edges: {global_dag_edges}")
            
            try:
                output_dir = self.load('output_dir')
                dag_viz = nx.DiGraph()
                dag_viz.add_edges_from(global_dag_edges)
                
                viz_filename = os.path.join(output_dir, f"participant_global_dag_iteration_{iteration}.png")
                participant = Client()
                participant.visualize_network(
                    dag_viz,
                    viz_filename,
                    target=target,
                    edge_weights=global_dag_weights
                )
                self.log(f"[PARTICIPANT] Global DAG visualization saved to: {viz_filename}")
            except Exception as e:
                self.log(f"[PARTICIPANT] Warning: Could not save DAG visualization: {e}")
        else:
            self.log(f"[PARTICIPANT] No global DAG edges received")

        if global_pam is not None:
            self.log(f"[PARTICIPANT] Received global PAM: {global_pam.shape}")
            
        should_stop = coordinator_payload.get('should_stop', False)
        
        if should_stop:
            self.log("[PARTICIPANT] Early stopping signal received - but first aggregate betas")
            return SEND_BETAS 
        else:
            self.log("[PARTICIPANT] Transitioning to SEND_BETAS for SMPC aggregation")
            return SEND_BETAS


@app_state(SEND_BETAS, Role.BOTH)
class SendBetasState(AppState):
    def register(self):
        self.register_transition(AGGREGATE_BETAS, Role.COORDINATOR)
        self.register_transition(AWAIT_AGGREGATED_BETAS, Role.PARTICIPANT)
    
    def run(self):
        iteration = self.load('iteration')
        homogeneous = self.load('homogeneous')
        
        if not homogeneous or iteration <= 1:
            self.log(f"[PARTICIPANT] Skipping beta SMPC (homogeneous={homogeneous}, iteration={iteration})")
            self.send_data_to_coordinator([], use_smpc=True)
            
            if self.is_coordinator:
                return AGGREGATE_BETAS
            else:
                return AWAIT_AGGREGATED_BETAS
        
        client_betas = self.load('client_betas_to_send')
        
        if client_betas is None:
            self.log(f"[PARTICIPANT] No beta parameters to send")
            self.send_data_to_coordinator([], use_smpc=True)
            
            if self.is_coordinator:
                return AGGREGATE_BETAS
            else:
                return AWAIT_AGGREGATED_BETAS
            
        node_order = self.load('node_order')
        param_positions = self.load('param_positions')
        total_params = self.load('total_params')
        
        self.log(f"[PARTICIPANT] Beta parameters structure:")
        for node, params in client_betas.items():
            self.log(f"  {node}: {list(params.keys())}")
            if 'intercept' in params:
                intercept_shape = np.array(params['intercept']).shape
                self.log(f"    intercept shape: {intercept_shape}")
            if 'coefficients' in params:
                coef_shape = np.array(params['coefficients']).shape
                self.log(f"    coefficients shape: {coef_shape}")
        
        self.log(f"[PARTICIPANT] Parameter positions:")
        for node in node_order:
            if node in param_positions:
                self.log(f"  {node}: {param_positions[node]}")
        
        coordinator = Coordinator()
        flat_vector = coordinator.flatten_betas(client_betas, node_order, param_positions, total_params)
        
        self.log(f"[PARTICIPANT] Flattened beta params to vector of length {len(flat_vector)}")
        self.log(f"[PARTICIPANT] First 5 values: {flat_vector[:5]}")
        
        weight = self.load('local_weight')
        
        if weight is None:
            self.log(f"[PARTICIPANT] ERROR: local_weight not found, using fallback")
            dataset_size = self.load('dataset_size')
            total_size = self.load('total_data_size')
            weight = dataset_size / total_size if total_size else 0.0
        
        weighted_vector = flat_vector * weight
        
        self.log(f"[PARTICIPANT] Using participant weight: {weight:.4f}")
        self.log(f"[PARTICIPANT] Weighted vector first 5: {weighted_vector[:5]}")
        
        # Send weighted vector via SMPC
        self.send_data_to_coordinator(weighted_vector.tolist(), use_smpc=True)
        
        if self.is_coordinator:
            self.log("[COORDINATOR] Sent weighted betas via SMPC -> AGGREGATE_BETAS")
            return AGGREGATE_BETAS
        else:
            self.log("[PARTICIPANT] Sent weighted betas via SMPC -> AWAIT_AGGREGATED_BETAS")
            return AWAIT_AGGREGATED_BETAS


@app_state(AGGREGATE_BETAS, Role.COORDINATOR)
class AggregateBetasState(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION_2, Role.COORDINATOR)
        self.register_transition(FINAL, Role.COORDINATOR)
    
    def run(self):
        iteration = self.load('iteration')
        homogeneous = self.load('homogeneous')
        should_stop = self.load('should_stop')

        aggregated_vector_raw = self.gather_data(use_smpc=True)
        
        if isinstance(aggregated_vector_raw, list) and len(aggregated_vector_raw) > 0:
            if isinstance(aggregated_vector_raw[0], list):
                aggregated_vector = aggregated_vector_raw[0]
            else:
                aggregated_vector = aggregated_vector_raw
        else:
            aggregated_vector = aggregated_vector_raw
        
        global_betas = None
        
        if homogeneous and iteration > 1 and len(aggregated_vector) > 0:
            self.log(f"[COORDINATOR] Received aggregated beta vector of length {len(aggregated_vector)}")
            self.log(f"[COORDINATOR] Aggregated vector first 5: {aggregated_vector[:5]}")
            
            node_order = self.load('node_order')
            param_positions = self.load('param_positions')
            beta_metadata = self.load('beta_metadata')
            
            coordinator = Coordinator()
            global_betas = coordinator.unflatten_betas(
                aggregated_vector, 
                node_order, 
                param_positions,
                beta_metadata
            )
            
            self.log(f"[COORDINATOR] Reconstructed global beta parameters:")
            for node, params in global_betas.items():
                self.log(f"  {node}:")
                if 'intercept' in params:
                    self.log(f"    intercept shape: {params['intercept'].shape}")
                if 'coefficients' in params:
                    self.log(f"    coefficients shape: {params['coefficients'].shape}")
                if 'parents' in params:
                    self.log(f"    parents: {params['parents']}")
                if 'classes' in params:
                    self.log(f"    classes: {params['classes']}")
            
            # Verify metadata is present
            missing_metadata = []
            for node in global_betas:
                if 'parents' not in global_betas[node] or 'classes' not in global_betas[node]:
                    missing_metadata.append(node)
            
            if missing_metadata:
                self.log(f"[COORDINATOR] WARNING: Missing metadata for nodes: {missing_metadata}")
            else:
                self.log(f"[COORDINATOR] All {len(global_betas)} nodes have complete metadata (parents and classes)")
            
            self.store("global_betas_local", global_betas)
        else:
            self.log(f"[COORDINATOR] No beta aggregation (homogeneous={homogeneous}, iteration={iteration})")
        
        beta_payload = {
            "global_betas": global_betas,
            "should_stop": should_stop,
            "Message": f"Iteration {iteration} betas aggregated"
        }
        
        bbytes = payload_size_bytes(beta_payload)
        self.log(f"[PAYLOAD] Coordinator->Clients beta payload size: {bbytes} bytes")
        
        self.broadcast_data(beta_payload)
        
        iteration += 1
        self.store('iteration', iteration)
        
        max_iterations = self.load('max_iterations')
        if should_stop:
            self.log("[COORDINATOR] Early stopping - waiting for clients to process and transition to FINAL")
            return FINAL
        elif iteration <= max_iterations:
            self.log("[COORDINATOR] Continuing to next iteration")
            return LOCAL_COMPUTATION_2
        else:
            self.log("[COORDINATOR] Max iterations reached - transitioning to FINAL")
            return FINAL


@app_state(AWAIT_AGGREGATED_BETAS, Role.PARTICIPANT)
class AwaitAggregatedBetasState(AppState):
    def register(self):
        self.register_transition(LOCAL_COMPUTATION_2, Role.PARTICIPANT)
        self.register_transition(FINAL, Role.PARTICIPANT)
    
    def run(self):
        iteration = self.load('iteration')
        beta_payload = self.await_data()
        
        bbytes = payload_size_bytes(beta_payload)
        self.log(f"[PAYLOAD] Coordinator->Client beta payload size: {bbytes} bytes")
        
        global_betas = beta_payload.get('global_betas')
        
        if global_betas is not None:
            self.log(f"[PARTICIPANT] Received global beta parameters:")
            for node, params in global_betas.items():
                self.log(f"  {node}:")
                if 'intercept' in params:
                    self.log(f"    intercept shape: {params['intercept'].shape}")
                if 'coefficients' in params:
                    self.log(f"    coefficients shape: {params['coefficients'].shape}")
            
            self.store('global_betas_local', global_betas)
        else:
            self.log(f"[PARTICIPANT] No global betas received")
        
        iteration += 1
        self.store('iteration', iteration)
        
        should_stop = beta_payload.get('should_stop', False)
        max_iterations = self.load('max_iterations')
        
        if should_stop:
            self.log("[PARTICIPANT] Early stopping - transitioning to FINAL")
            return FINAL
        elif iteration <= max_iterations:
            self.log("[PARTICIPANT] Continuing to next iteration")
            return LOCAL_COMPUTATION_2
        else:
            self.log("[PARTICIPANT] Max iterations reached - transitioning to FINAL")
            return FINAL


@app_state(FINAL, Role.BOTH)
class FinalState(AppState):
    def register(self):
        self.register_transition(TERMINAL, Role.BOTH)

    def run(self):
        output_dir = self.load('output_dir')
        self.log("="*60)
        self.log("FINAL STATE - Generating outputs and summaries")
        self.log("="*60)
        
        target = self.load('target')
        
        participant = Client()
        
        try:
            metrics_history = self.load('metrics_history')
            
            self.log("Training Summary:")
            self.log("="*60)
            self.log(f"Total Iterations: {len(metrics_history['iteration'])}")
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

        try:
            homogeneous = self.load('homogeneous')
            dataset = self.load('dataset')
            export_payload = None
            export_path = None
            
            if homogeneous:
                
                global_dag_edges = self.load('global_dag_edges_local') or []
                global_params = self.load('global_betas_local')
                
                
                if (global_params is None or len(global_params) == 0) and dataset is not None:
                    global_dag = nx.DiGraph()
                    global_dag.add_edges_from(global_dag_edges)
                    if dataset is not None:
                        global_dag.add_nodes_from(dataset.columns.tolist())
                    global_params = participant.estimate_multilogit_params(global_dag, dataset)
                
                export_payload = _build_network_export_payload(
                    dag_edges=global_dag_edges,
                    params=global_params,
                    dataset=dataset,
                    participant=participant
                )
                export_path = os.path.join(output_dir, 'final_global_network_params.json')
            else:
                
                local_dag = self.load('final_dag')
                local_params = self.load('local_params')
                
                if local_dag is None:
                    local_dag = nx.DiGraph()
                if dataset is not None and local_dag.number_of_nodes() == 0:
                    local_dag.add_nodes_from(dataset.columns.tolist())
                
                if (local_params is None or len(local_params) == 0) and dataset is not None:
                    local_params = participant.estimate_multilogit_params(local_dag, dataset)
                
                export_payload = _build_network_export_payload(
                    dag_edges=list(local_dag.edges()),
                    params=local_params,
                    dataset=dataset,
                    participant=participant
                )
                export_path = os.path.join(output_dir, 'final_local_network_params.json')
            
            if export_payload is not None and export_path is not None:
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(export_payload, f, default=_json_default, ensure_ascii=False, indent=2)
                self.log(f"Final network+parameters JSON saved to: {export_path}")
        except Exception as e:
            self.log(f"Warning: Could not save final network+parameters JSON: {e}")
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