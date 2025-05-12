import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed
import numpy as np
import tqdm


class FQI:
    def __init__(self, n_actions, regressor=None, gamma=0.99, n_iterations=50):
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_iterations = n_iterations
        # Un regressore per ogni azione
        if regressor is None:
            self.regressors = [HistGradientBoostingRegressor() for _ in range(n_actions)]
        else:
            self.regressors = [regressor() for _ in range(n_actions)]

    def fit(self, transitions):
        """
        transitions: lista di tuple (state, action, reward, next_state, done)
        """
        X = np.array([np.hstack([t[0], t[1]]) for t in transitions])
        y = np.zeros(len(transitions))
        for a in range(self.n_actions):
            idx = X[:, -1] == a
            if np.any(idx):
                self.regressors[a].fit(X[idx], y[idx])

        for _ in range(self.n_iterations):
            # Calcola Q per ogni azione nel next_state
            Q_next = np.zeros((len(transitions), self.n_actions))
            for a in range(self.n_actions):
                next_states = np.array([t[3] for t in transitions])
                actions = np.full((len(transitions), 1), a)
                X_next = np.hstack([next_states, actions])
                Q_next[:, a] = self.regressors[a].predict(X_next)
            # Aggiorna target y
            y = np.array([
                t[2] + self.gamma * np.max(Q_next[i]) * (not t[4])
                for i, t in enumerate(transitions)
            ])
            # Fit per ogni azione
            for a in range(self.n_actions):
                idx = X[:, -1] == a
                if np.any(idx):
                    self.regressors[a].fit(X[idx], y[idx])

    def predict(self, state):
        """
        Restituisce l'azione greedy per lo stato dato.
        """
        q_values = []
        for a in range(self.n_actions):
            q = self.regressors[a].predict(np.hstack([state, a]).reshape(1, -1))
            q_values.append(q)
        return int(np.argmax(q_values))