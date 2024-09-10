import strategy
from tqdm import tqdm
import argparse
from functools import partial
import numpy as np
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--theta_x", type=float, default=0.2)
    parser.add_argument("--theta_y", type=float, default=0.3)
    parser.add_argument("--theta_0", type=float, default=0.1)
    parser.add_argument("--theta_1", type=float, default=0.4)
    parser.add_argument("--repeat", type=int, default=20000)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--M", type=int, default=30)
    parser.add_argument("--beta_0", type=float, default=10, help="hyper parameter beta_0")
    parser.add_argument("--alpha_xy", type=float, default=10, help="hyper parameter alpha_xy")

    args = parser.parse_args()

    pbar = tqdm(range(args.repeat))
    print(args)

    strategies = {
        "Random": strategy.Random,
        "MaxPass": strategy.MaxPass,
        "CodeT": strategy.CodeT,
        "B4": partial(strategy.B4, beta_0=args.beta_0, alpha_xy=args.alpha_xy),
    }
    result_acc_list = {}
    for i in pbar:
        
        true_x = (np.random.random((args.N, )) < args.theta_x).astype(np.int32).reshape((-1, 1))
        true_y = (np.random.random((args.M, )) < args.theta_y).astype(np.int32).reshape((1, -1))

        edge_prob = true_x * true_y + (1 - true_x) * true_y * args.theta_1 + (1 - true_x) * (1 - true_y) * args.theta_0
        edge = (np.random.random((args.N, args.M)) < edge_prob).astype(np.int32)
        if np.sum(true_x) == 0 or np.sum(true_x) == args.N:
            continue

        result = []
        for cur_strategy_name, cur_strategy in list(strategies.items()):
            if cur_strategy_name not in result_acc_list:
                result_acc_list[cur_strategy_name] = []
            st = time.time()
            for _ in range(1):
                pred_x_loss = cur_strategy(edge)
            t = (time.time() - st) / 1
            best_pred_x = pred_x_loss[0][0]
            pred_x = np.zeros(args.N)
            for x in best_pred_x:
                pred_x[x] = 1
            # acc = np.mean(true_x.reshape((-1, ))[pred_x == 1])
            acc = t
            result_acc_list[cur_strategy_name].append(acc)
            acc = np.mean(result_acc_list[cur_strategy_name])
            result.append(f"{cur_strategy_name}: {acc:.10f}")

        if i % 100:
            pbar.set_description(' | '.join(result))