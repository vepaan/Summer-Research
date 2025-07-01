import numpy as np
import stormpy
import stormpy.core
import stormpy.builder as builder

def compute_win_probability(desc: np.ndarray, slip: list[float]) -> float:
    size = desc.shape[0]
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

    builder_options = builder.SparseModelBuilderOptions()
    model_builder = builder.SparseModelBuilder(builder_options, stormpy.storage.ModelType.MDP)
    model_builder.set_number_of_states(num_states)

    # Initial state is top-left
    initial_state = to_state(0, 0)
    model_builder.set_initial_state(initial_state)

    for i in range(size):
        for j in range(size):
            s = to_state(i, j)
            tile = desc[i][j]

            if tile == b'H' or tile == b'G':
                # Terminal states (self-loop)
                model_builder.add_transition(s, s, 1.0)
                continue

            intended_action = 1  # DOWN (as before)
            deltas = slip_deltas[intended_action]

            for (dx, dy), prob in zip(deltas, slip):
                ni, nj = i + dx, j + dy
                if 0 <= ni < size and 0 <= nj < size:
                    s_next = to_state(ni, nj)
                else:
                    s_next = s  # Wall → stay in same state
                model_builder.add_transition(s, s_next, prob)

    model = model_builder.build()

    # Goal is bottom-right corner
    goal_state = to_state(size - 1, size - 1)

    # Define property: Pmax=? [ F goal ]
    formula_str = f'Pmax=? [F s={goal_state}]'
    formulas = stormpy.parse_properties(formula_str, model)

    result = stormpy.model_checking(model, formulas[0])
    return result.at(initial_state)


def approximate_win_probability(desc: np.ndarray, slip: list[float]) -> float:
    size = desc.shape[0]
    hole_ratio = np.sum(desc == b'H') / (size * size)
    slip_penalty = slip[1] + slip[2]  # penalty for lateral slips
    return max(0.0, min(1.0, 1.0 - (hole_ratio + 0.5 * slip_penalty) ** 1.5))
