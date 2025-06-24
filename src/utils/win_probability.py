import numpy as np

def compute_win_probability(desc: np.ndarray, size: int, is_slippery: bool):

    num_states = size * size

    def to_state(i, j):
        return i * size + j

    dir_list = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    P = np.zeros((num_states, num_states), dtype=np.float64)

    for i in range(size):
        for j in range(size):
            s = to_state(i, j)
            tile = desc[i][j]

            if tile == b'H' or tile == b'G':
                # Absorbing state: probability 1 to stay here
                P[s, s] = 1.0
                continue

            transitions = {}

            for dx, dy in dir_list:
                slips = []
                if is_slippery:
                    slips = [
                        (dx, dy),
                        (-dy, dx),  # left turn
                        (dy, -dx),  # right turn
                    ]
                    prob = 1.0 / 3.0
                else:
                    slips = [(dx, dy)]
                    prob = 1.0  # deterministic movement, probability 1

                for Dx, Dy in slips:
                    ni, nj = i + Dx, j + Dy
                    if 0 <= ni < size and 0 <= nj < size:
                        s_next = to_state(ni, nj)
                    else:
                        s_next = s  # hit wall, stay in place

                    transitions[s_next] = transitions.get(s_next, 0.0) + prob

            # Normalize to sum 1 (usually already normalized)
            total = sum(transitions.values())
            for k in transitions:
                transitions[k] /= total

            for k, p in transitions.items():
                P[s, k] = p

    # Now solve reachability probability vector x with:
    # x[s] = 1 if s is goal
    # x[s] = sum over s' P[s,s'] * x[s'] otherwise

    goal_state = to_state(size - 1, size - 1)

    x = np.zeros(num_states, dtype=np.float64)
    x[goal_state] = 1.0

    eps = 1e-12
    max_iter = 10000

    for iteration in range(max_iter):
        x_new = np.zeros_like(x)
        for s in range(num_states):
            tile = desc[s // size][s % size]
            if tile == b'G':
                x_new[s] = 1.0
            elif tile == b'H':
                x_new[s] = 0.0
            else:
                x_new[s] = np.dot(P[s], x)
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            break
        x = x_new

    return x[0]  # probability starting from initial state (0,0)
