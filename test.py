from typing import List


def levenshtein_distance(pred: str, gt: str) -> int:
    """
    Calculates the levenshtein distance between two words.
    See: http://www.orand.cl/en/icfhr2014-hdsr/#evaluation
    :param pred: the predicted word
    :param gt: the ground truth
    :return: the levenshtein distance
    """
    raise NotImplementedError()


def normalized_levenshtein_distance(pred: str, gt: str) -> float:
    """
    Calculates the normalzed levenshtein distance between two words.
    See: http://www.orand.cl/en/icfhr2014-hdsr/#evaluation
    :param pred: the predicted word
    :param gt: the ground truth
    :return: the normalized levenshtein distance
    """
    raise NotImplementedError()


def average_nomralized_levenshtein_distance(preds: List[str], gt: List[str]) -> float:
    """
    Calculates the average normalized levenshtein distance on a dataset.
    See: http://www.orand.cl/en/icfhr2014-hdsr/#evaluation
    :param preds: the predicted words
    :param gt: the ground truth in the same order
    :return: the average normalized levenshtein distance
    """
    raise NotImplementedError()


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
    raise NotImplementedError()
