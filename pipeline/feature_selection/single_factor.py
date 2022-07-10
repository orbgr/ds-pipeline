from sklearn.metrics import roc_auc_score


def gini(X, y):
    func = lambda x, y: abs(2 * roc_auc_score(y, x) - 1)
    res = X.apply(lambda x: func(x, y), axis=0)
    return res


def coverage(X, y=None):
    return 1 - X.isnull().mean()


def uniq(X, y=None):
    return X.value_counts(dropna=False, normalize=True).max()


def correlation(X, y):
    func = lambda x, y: abs(x.corr(y))
    res = X.apply(lambda x: func(x, y), axis=0)
    return res