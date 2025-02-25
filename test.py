from typing import List, Any

import numpy as np


def levenshtein_distance(pred: List[Any], gt: List[Any], w_sub=1, w_del=1, w_ins=1) -> int:
    """
    Calculates the levenshtein distance between two words.
    See: http://www.orand.cl/en/icfhr2014-hdsr/#evaluation
    :param pred: the predicted word
    :param gt: the ground truth
    :param w_sub: weight for substitutions
    :param w_del: weight for deletions
    :param w_ins: weight for insertions
    :return: the levenshtein distance
    """
    m = len(gt)
    n = len(pred)

    loc_pred = [" "] + pred
    loc_gt = [" "] + gt

    d = np.zeros((n+1, m+1),dtype=np.int)

    for i in range(0, m+1):
        d[0][i] = w_ins*i

    for j in range(0, n+1):
        d[j][0] = w_del*j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if loc_gt[j] == loc_pred[i]:
                d[i][j] = d[i-1][j-1]

            else:
                ins = d[i][j-1] + w_ins
                rem = d[i-1][j] + w_del
                sub = d[i-1][j-1] + w_sub

                d[i][j] = min(ins, rem, sub)

    return d[n][m]


def normalized_levenshtein_distance(pred: str, gt: str, w_sub=1, w_del=1, w_ins=1) -> float:
    """
    Calculates the normalzed levenshtein distance between two words.
    See: http://www.orand.cl/en/icfhr2014-hdsr/#evaluation
    :param pred: the predicted word
    :param gt: the ground truth
    :param w_sub: weight for substitutions
    :param w_del: weight for deletions
    :param w_ins: weight for insertions
    :return: the normalized levenshtein distance
    """
    ld = levenshtein_distance(pred, gt, w_sub, w_del, w_ins)
    return np.min(ld, len(gt))/len(gt)


def average_normalized_levenshtein_distance(preds: List[str], gt: List[str], w_sub=1, w_del=1, w_ins=1) -> float:
    """
    Calculates the average normalized levenshtein distance on a dataset.
    See: http://www.orand.cl/en/icfhr2014-hdsr/#evaluation
    :param preds: the predicted words
    :param gt: the ground truth in the same order
    :return: the average normalized levenshtein distance
    """
    nld = np.sum([levenshtein_distance(pred, target, w_sub, w_del, w_ins) for pred, target in zip(preds, gt)])
    return nld/len(gt)


def recognition_rate(preds: List[List[str]], gt: List[str], top_k=3):
    """
    Calculates the recognition rate given the top k predictions
    for each datapoint and the ground truth.
    See: http://www.orand.cl/en/icfhr2014-hdsr/#evaluation
    :param preds: the predictions as a list of lists (of length top_k) of strings
    :param gt: the ground truth in the same order
    :param top_k: the number of allowed predictions
    :return: the recognition rate
    """
    recog = 0

    for k_preds, target in zip(preds, gt):
        for pred in k_preds.sort()[:-top_k]:
            if pred == target:
                recog += 1
                continue

    return recog/float(len(gt))
