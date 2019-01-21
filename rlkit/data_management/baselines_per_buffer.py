from baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer

class BaselinesPERBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            alpha,
            ):
        self.underlying = PrioritizedReplayBuffer(max_replay_buffer_size, alpha)
        
    def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
        self.underlying.add(observation, action, reward, next_observation, terminal)

    def random_batch(self, batch_size, beta):
        return self.underlying.sample(batch_size, beta)

    def num_steps_can_sample(self):
        return len(self.underlying)

    def update_priorities(self, *args, **kwargs):
        self.underlying.update_priorities(*args, **kwargs)
