import numpy as np

def compute_win_probability(desc: np.ndarray, size: int, slip: list[float]) -> float:
    num_states = size * size

    def to_state(i, j):
        return i * size + j

    move = {
        0: (0, -1),  # LEFT
        1: (1, 0),   # DOWN
        2: (0, 1),   # RIGHT
        3: (-1, 0),  # UP
    }

    slip_deltas = {
        0: [move[0], move[3], move[1]],  # LEFT → [LEFT, UP, DOWN]
        1: [move[1], move[0], move[2]],  # DOWN → [DOWN, LEFT, RIGHT]
        2: [move[2], move[1], move[3]],  # RIGHT → [RIGHT, DOWN, UP]
        3: [move[3], move[2], move[0]],  # UP → [UP, RIGHT, LEFT]
    }

    P = np.zeros((num_states, num_states), dtype=np.float64)

    for i in range(size):
        for j in range(size):
            s = to_state(i, j)
            tile = desc[i][j]

            if tile == b'H' or tile == b'G':
                P[s, s] = 1.0
                continue

            # Choose one fixed action direction — for now assume DOWN (can generalize later)
            intended_action = 1  # assume DOWN (you can parameterize this)

            for (dx, dy), prob in zip(slip_deltas[intended_action], slip):
                ni, nj = i + dx, j + dy
                if 0 <= ni < size and 0 <= nj < size:
                    s_next = to_state(ni, nj)
                else:
                    s_next = s  # hit wall → stay

                P[s, s_next] += prob

    # Solve: x[s] = sum P[s, s'] * x[s'] with x[goal] = 1
    goal_state = to_state(size - 1, size - 1)
    x = np.zeros(num_states, dtype=np.float64)
    x[goal_state] = 1.0

    eps = 1e-12
    max_iter = 10000
    for _ in range(max_iter):
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

    return x[0]
