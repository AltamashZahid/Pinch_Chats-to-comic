"""Grey Wolf Optimizer (Mirjalili et al., 2014), maximising a black-box function.

Wolves explore a box; each step they move toward the three best found so far (alpha, beta,
delta), with an exploration coefficient `a` shrinking from 2 to 0. Used where the objective
has no gradient (it goes through argmax / image sampling): per-emotion logit biases for the
text classifier, and per-emotion enhancer strengths for the image backend.
"""
import numpy as np


def gwo(f, lo, hi, wolves: int = 12, iters: int = 40, seed: int = 0, init=None, log=None):
    """Maximise f(x) over lo <= x <= hi. Returns (best_x, best_value, history)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    X = rng.uniform(lo, hi, (wolves, len(lo)))
    if init is not None:
        X[0] = init  # e.g. the current setting, so the result is never worse than it
    fit = np.array([f(x) for x in X])
    history = [float(fit.max())]
    for t in range(iters):
        order = np.argsort(-fit)
        leaders = X[order[:3]].copy()
        a = 2 - 2 * t / iters
        for i in range(wolves):
            moves = []
            for L in leaders:
                A = 2 * a * rng.random(len(lo)) - a
                C = 2 * rng.random(len(lo))
                moves.append(L - A * np.abs(C * L - X[i]))
            X[i] = np.clip(np.mean(moves, axis=0), lo, hi)
            fit[i] = f(X[i])
        history.append(float(fit.max()))
        if log:
            log(t, float(fit.max()))
    best = int(np.argmax(fit))
    return X[best].copy(), float(fit[best]), history
