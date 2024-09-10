import argparse
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import random
from sklearn.metrics import precision_score
from functools import partial

from src.postprocess import PostProcessor
from src.execution import evaluate_with_test_code, evaluate_with_test_cases
from src.io_utils import Tools
from src.agreement import DataManager, DualAgreement
from collections import defaultdict
from tqdm import tqdm
from src.evaluation import pass_at_K, get_result_of_sorted_solutions, _dictionized_ground_truth_results
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import strategy
import json
import hashlib

import warnings
warnings.filterwarnings("ignore")
from scipy.stats import wilcoxon
import copy

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    configs = json.load(open("data/config.json"))
    cachable = True

    parser.add_argument("--config_name", type=str, help="config name in data/config.json. By default, run all configs", default='')
    parser.add_argument("--cache_dir", type=str, help="the directory to store the cache files", default='cache')
    parser.add_argument("--timeout", type=float, default=0.1, help="how many seconds to wait during execution for each test case")
    parser.add_argument("--max_px", type=float, default=1.0, help="max probability of correct code")
    parser.add_argument("--beta_0_range", type=str, default="1e4,1e5,1e6", help="hyper parameter beta_0 range")
    parser.add_argument("--alpha_xy_range", type=str, default="1e3", help="hyper parameter alpha_xy range")

    args = parser.parse_args()

    strategies = {
        "Random": strategy.Random,
        "MaxPass": strategy.MaxPass,
        "MBR_exec": strategy.MBR_exec,
        "CodeT": strategy.CodeT
    }
    beta_0_range = list(map(float, args.beta_0_range.split(",")))
    alpha_xy_range = list(map(float, args.alpha_xy_range.split(",")))
    for beta_0 in beta_0_range:
        for alpha_xy in alpha_xy_range:
            strategies[f'B({beta_0:.1e}, {alpha_xy:.1e})'] = partial(strategy.B4, beta_0=beta_0, alpha_xy=alpha_xy)

    os.makedirs("cache", exist_ok=True)

    for key in configs:
        if args.config_name != '' and key not in args.config_name:
            continue

        flavor = configs[key]['flavor']
        source_path_for_solution = configs[key]['dataset_code']
        predict_path_for_solution = configs[key]['generated_code']
        source_path_for_test = configs[key]['dataset_test']
        predict_path_for_test = configs[key]['generated_test']
    
        handled_solutions = PostProcessor.map_task_id_for_solution(predict_path_for_solution, source_path_for_solution)
        handled_test_cases = PostProcessor.map_task_id_for_test_case(predict_path_for_test, source_path_for_test)

        cache_hash = key
        
        cache_ground_truth_exec_result = os.path.join(args.cache_dir, f'ground_truth_exec_result_{cache_hash}.pkl')
        if os.path.exists(cache_ground_truth_exec_result) and cachable:
            ground_truth_exec_result = Tools.load_pickle(cache_ground_truth_exec_result)
        else:
            ground_truth_exec_result = evaluate_with_test_code(handled_solutions, timeout=args.timeout)
            Tools.dump_pickle(cache_ground_truth_exec_result, ground_truth_exec_result)

        cache_dual_exec_result = os.path.join(args.cache_dir, f'dual_exec_result_{cache_hash}.pkl')

        if os.path.exists(cache_dual_exec_result) and cachable:
            dual_exec_result = Tools.load_pickle(cache_dual_exec_result)
        else:
            dual_exec_result = evaluate_with_test_cases(handled_solutions, handled_test_cases, timeout=args.timeout)
            Tools.dump_pickle(cache_dual_exec_result, dual_exec_result)
    
        if flavor != "":    
            valid_task_id = [x['task_id'] for x in Tools.load_jsonl(source_path_for_test) if x['meta_data']['difficulty'] == flavor]
            dual_exec_result = [x for x in dual_exec_result if x['task_id'] in valid_task_id]
            handled_solutions = [x for x in handled_solutions if x['task_id'] in valid_task_id]
            ground_truth_exec_result = [x for x in ground_truth_exec_result if x['task_id'] in valid_task_id]
            handled_test_cases_filtered = defaultdict()
            for x in handled_test_cases:
                if x in valid_task_id:
                    handled_test_cases_filtered[x] = handled_test_cases[x]
            handled_test_cases = handled_test_cases_filtered
            print(len(dual_exec_result), len(handled_solutions), len(ground_truth_exec_result), len(handled_test_cases))

        ground_truth_results_by_task_and_solution = defaultdict(defaultdict)
        for result in ground_truth_exec_result:
            ground_truth_results_by_task_and_solution[result['task_id']][result['completion']] = result['passed']

        task_id_to_passing_matrix = defaultdict(lambda: [])
        task_id_to_code_list = defaultdict(lambda: [])
        task_id_to_code_correctness = defaultdict(lambda: [])

        for sample in dual_exec_result:
            if not sample['passed']: 
                continue
            
            task_id_to_code_list[sample['task_id']].append(sample['completion'])
            task_id_to_code_correctness[sample['task_id']].append(ground_truth_results_by_task_and_solution[sample['task_id']][sample['completion']])
            current_code_E = []
            for t, r in zip(sample['test_cases'], sample['result']):
                current_code_E.append(int(r))
            task_id_to_passing_matrix[sample['task_id']].append(current_code_E)
        
        valid_task_id = set()
        invalid_task_id = set()
        for task_id in task_id_to_code_correctness:
            pass_rate = np.mean(task_id_to_code_correctness[task_id])
            if pass_rate != 0.0 and pass_rate < args.max_px:
                valid_task_id.add(task_id)
            else:
                invalid_task_id.add(task_id)
        for task_id in invalid_task_id:
            del task_id_to_code_correctness[task_id]
            del task_id_to_passing_matrix[task_id]
            del task_id_to_code_list[task_id]
        ground_truth_exec_result = [x for x in ground_truth_exec_result if x['task_id'] in valid_task_id]

        for task in task_id_to_passing_matrix:
            E = np.asarray(task_id_to_passing_matrix[task]).astype(np.int8)
            true_x = np.asarray(task_id_to_code_correctness[task]).astype(np.int8)

        method_acc = []
        for cur_strategy_name, cur_strategy in tqdm(list(strategies.items())):
            result = {}
            for task_id in task_id_to_passing_matrix:
                E = np.asarray(task_id_to_passing_matrix[task_id]).astype(np.int8)
                true_x = np.asarray(task_id_to_code_correctness[task_id]).astype(np.int8)
                sorted_list = cur_strategy(E)
                cur_result = []
                for solution_ids, score in sorted_list:
                    solutions = list(map(lambda x: task_id_to_code_list[task_id][x], solution_ids))
                    cur_result.append((solutions, score))
                result[task_id] = cur_result
            acc = get_result_of_sorted_solutions(ground_truth_exec_result, result)['pass@1']
            method_acc.append(f"{cur_strategy_name}: {acc}")
        
        with open(os.path.join(args.cache_dir, f'result_{cache_hash}.json'), "w") as f:
            json.dump(method_acc, f, indent=2)
        print("==================")
        print(key)
        print("\n".join(method_acc))