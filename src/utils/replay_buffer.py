import random
from collections import deque, namedtuple

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:

    def __init__(self, buffer_size: int):
        self.memory = deque([], maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.memory, k=batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)