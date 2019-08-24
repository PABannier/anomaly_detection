import numpy as np

class Detector:
    def __init__(self, X_train, X_val, y_val):
        self.X_train = X_train
        self.X_val, self.y_val = X_val, y_val
        self.m, self.n = X_val.shape

    def fit(self):
        self.mu, self.sigma = self.__compute_mean(self.X_train), self.__compute_std(self.X_train)
        proba = np.zeros(self.m)

        for i in range(self.m):
            p = self.__compute_gaussian(self.X_val[i,:], self.mu, self.sigma, self.n)
            proba[i] = np.prod(p)

        self.epsilon = self.__find_epsilon(proba, self.y_val)

    def predict(self, X):
        m1, n1 = X.shape
        proba = np.zeros(m1)

        for i in range(m1):
            p = self.__compute_gaussian(X[i,:], self.mu, self.sigma, n1)
            proba[i] = np.prod(p)

        return proba < self.epsilon

    def __find_epsilon(self, proba, y_true):
        best_epsilon = 0
        best_f1_score = 0

        for epsilon in np.arange(np.min(proba), np.max(proba), (np.max(proba) - np.min(proba))/1000):
            y_pred = (proba < epsilon)[:, np.newaxis]
            f1_score = self.__compute_f1_score(np.expand_dims(y_pred, axis=1), y_true)

            if f1_score >= best_f1_score:
                best_epsilon = epsilon
                best_f1_score = f1_score

        return best_epsilon

    @staticmethod
    def __compute_gaussian(x, mu, sigma, n):

        if n > 1:
            sigma = np.diag(sigma)

        part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(sigma) ** (1 / 2)))
        part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(sigma))).dot((x - mu))

        return float(part1 * np.exp(part2))

    @staticmethod
    def __compute_mean(X):
        return np.mean(X, 0)

    @staticmethod
    def __compute_std(X):
        return np.std(X, 0)

    @staticmethod
    def __compute_f1_score(y_pred, y_true):

        tp = np.sum(y_pred[y_true == 1] == 1)
        fp = np.sum(y_pred[y_true == 0] == 1)
        fn = np.sum(y_pred[y_true == 1] == 0)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return (2 * precision * recall) / (precision + recall)
