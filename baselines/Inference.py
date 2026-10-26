import os
import logging

import numpy as np
import torch

from utils import utils

EVALUATION_RECORDS = {}


def _record_evaluation(index, phase, counts, values):
    metrics = {f'{name}@{cutoff}': float(values[position][offset]) for position, name in enumerate(('Recall', 'NDCG')) for offset, cutoff in enumerate((10, 20))}
    EVALUATION_RECORDS[index, phase] = {**counts, 'metrics': metrics}
    return values


def computeTopNAccuracy(GroundTruth, predictedIndices, topN, n_items, device):
    predicted = predictedIndices.detach().cpu().numpy() if torch.is_tensor(predictedIndices) else np.asarray(predictedIndices)
    targets = [set(int(item) for item in items) for items in GroundTruth]
    lengths = np.asarray([len(items) for items in targets], dtype=np.int64)
    valid = lengths > 0
    if not np.any(valid):
        return ([0.0] * len(topN), [0.0] * len(topN), [0.0] * len(topN), [0.0] * len(topN))
    hits = np.asarray([
        [int(item) >= 0 and int(item) in truth for item in row]
        for row, truth in zip(predicted, targets)
    ], dtype=np.float64)
    recalls, ndcgs, mrrs, precisions = [], [], [], []
    for K in topN:
        width = min(K, hits.shape[1])
        selected = hits[:, :width]
        count = selected.sum(axis=1)
        discounts = 1.0 / np.log2(np.arange(2, K + 2, dtype=np.float64))
        ideal = np.concatenate(([0.0], np.cumsum(discounts)))[np.minimum(lengths, K)]
        dcg = (selected * discounts[:width]).sum(axis=1)
        reciprocal = np.zeros(len(targets))
        if width:
            first = selected.argmax(axis=1)
            reciprocal[count > 0] = 1.0 / (first[count > 0] + 1)
        recalls.append(float(np.mean(count[valid] / lengths[valid])))
        precisions.append(float(np.mean(count[valid] / K)))
        ndcgs.append(float(np.mean(dcg[valid] / ideal[valid])))
        mrrs.append(float(np.mean(reciprocal[valid])))
    return (recalls, ndcgs, mrrs, precisions)

def Test(args, model, corpus, data_type, data_idx):
    batch_size = args.batch_size
    test_file = os.path.join(corpus.snapshots_path, f'{data_type}_block{data_idx}')
    test_data = np.asarray(utils.read_data_from_file_int(test_file), dtype=np.int64).reshape(-1, 3)
    test_pos = {}
    for (u, i, _) in test_data:
        test_pos.setdefault(int(u), set()).add(int(i))
    hist_file = os.path.join(corpus.snapshots_path, f'hist_block{data_idx}')
    hist_data = np.asarray(utils.read_data_from_file_int(hist_file), dtype=np.int64).reshape(-1, 3)
    hist_pos = {}
    for (u, i, _) in hist_data:
        hist_pos.setdefault(int(u), set()).add(int(i))
    filtered = {user: truth - hist_pos.get(user, set()) for user, truth in test_pos.items()}
    eligible = {user: truth for user, truth in filtered.items() if truth}
    counts = {'evaluated_users': len(eligible), 'excluded_users': len(test_pos) - len(eligible), 'target_users': len(test_pos), 'masked_targets': sum(len(test_pos[user]) - len(filtered[user]) for user in test_pos)}
    logging.info('Evaluation window %d %s: users=%d eligible_users=%d excluded_users=%d masked_targets=%d', data_idx + 1, data_type, len(test_pos), len(eligible), len(test_pos) - len(eligible), sum(len(test_pos[user]) - len(filtered[user]) for user in test_pos))
    Ks = [10, 20, 50, 100]
    max_K = min(max(Ks), corpus.n_items)
    users = sorted(eligible)
    if not users:
        return _record_evaluation(data_idx, data_type, counts, ([0.0] * len(Ks), [0.0] * len(Ks), [0.0] * len(Ks), [0.0] * len(Ks)))
    n_batch = (len(users) + batch_size - 1) // batch_size
    item_id = torch.arange(corpus.n_items, device=model._device)
    rating_list = []
    ground_truth_list = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        all_users, all_items = model.computer()
        for i in range(n_batch):
            batch_users = users[i * batch_size:(i + 1) * batch_size]
            user_id = torch.tensor(batch_users, dtype=torch.int64, device=model._device)
            scores = all_users[user_id] @ all_items[item_id].T
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Recommendation scores must be finite.')
            for (row, u) in enumerate(batch_users):
                if u in hist_pos:
                    scores[row, sorted(hist_pos[u])] = -torch.inf
            topk_items = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :max_K]
            topk_scores = scores.gather(1, topk_items)
            topk_items = topk_items.masked_fill(~torch.isfinite(topk_scores), -1)
            rating_list.append(topk_items.cpu())
            gt = [eligible[u] for u in batch_users]
            ground_truth_list.extend(gt)
    model.train(was_training)
    rating_list = torch.cat(rating_list, dim=0)
    (recall, ndcg, mrr, precision) = computeTopNAccuracy(ground_truth_list, rating_list, Ks, corpus.n_items, model._device)
    return _record_evaluation(data_idx, data_type, counts, (recall, ndcg, mrr, precision))

def print_results(loss, valid_result, test_result):
    result_str = ''
    if loss is not None:
        logging.info('[Train]: loss: {:.4f}'.format(loss))
    if valid_result is not None:
        logging.info('[Valid]: Recall: {} NDCG: {} MRR: {} Precision: {}'.format('-'.join([f'{x:.4f}' for x in valid_result[0]]), '-'.join([f'{x:.4f}' for x in valid_result[1]]), '-'.join([f'{x:.4f}' for x in valid_result[2]]), '-'.join([f'{x:.4f}' for x in valid_result[3]])))
        result_str += 'Recall@10\t' + str(valid_result[0][0]) + '\n'
        result_str += 'NDCG@10\t' + str(valid_result[1][0]) + '\n'
        result_str += 'Recall@20\t' + str(valid_result[0][1]) + '\n'
        result_str += 'NDCG@20\t' + str(valid_result[1][1]) + '\n'
        result_str += 'Recall@50\t' + str(valid_result[0][2]) + '\n'
        result_str += 'NDCG@50\t' + str(valid_result[1][2]) + '\n'
    if test_result is not None:
        logging.info('[Test]: Recall: {} NDCG: {} MRR: {} Precision: {}'.format('-'.join([f'{x:.4f}' for x in test_result[0]]), '-'.join([f'{x:.4f}' for x in test_result[1]]), '-'.join([f'{x:.4f}' for x in test_result[2]]), '-'.join([f'{x:.4f}' for x in test_result[3]])))
        result_str += 'Recall@10\t' + str(test_result[0][0]) + '\n'
        result_str += 'NDCG@10\t' + str(test_result[1][0]) + '\n'
        result_str += 'Recall@20\t' + str(test_result[0][1]) + '\n'
        result_str += 'NDCG@20\t' + str(test_result[1][1]) + '\n'
        result_str += 'Recall@50\t' + str(test_result[0][2]) + '\n'
        result_str += 'NDCG@50\t' + str(test_result[1][2]) + '\n'
    return result_str
