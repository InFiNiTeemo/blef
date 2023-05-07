from sklearn.metrics import f1_score
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

def get_f1(y_true, y_pred, threshold=0.5):
    return f1_score(y_true, y_pred>threshold, average='macro')


def padded_cmap(solution, submission, padding_factor=5):
    """
    solution, submission - 2d ndarray
    """
    solution = pd.DataFrame(solution)#.drop(['row_id'], axis=1, errors='ignore')
    submission = pd.DataFrame(submission)#.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score