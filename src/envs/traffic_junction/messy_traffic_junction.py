import numpy as np
from .traffic_junction import Traffic_JunctionEnv
from utils.dict2namedtuple import convert

class Messy_Traffic_JunctionEnv(Traffic_JunctionEnv):
    """
    A noisy, "messy" version of the TrafficJunction environment.
    Adds two types of randomness:
      1) Random initial steps to scramble starting state.
      2) With probability `failure_obs_prob`, flips an agent's observation.
    """
    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.init_random_steps = getattr(args, "randomize_initial_state", 10)
        self.failure_obs_prob = getattr(args, "failure_obs_prob", 0.15)
        args = args._asdict()
        args.pop("randomize_initial_state", 10)
        args.pop("failure_obs_prob", 0.15)
        super(Messy_Traffic_JunctionEnv, self).__init__(**args)

    def get_obs(self):
        """
        Return a list of observations, each possibly corrupted.
        """
        base_obs = super(Messy_Traffic_JunctionEnv, self).get_obs()
        noisy = []
        for obs in base_obs:
            if np.random.rand() <= self.failure_obs_prob:
                noisy.append(-1.0 * obs)
            else:
                noisy.append(obs)
        return noisy

    def get_obs_agent(self, agent_id: int):
        """
        Return a single agent's observation, possibly corrupted.
        """
        obs = super(Messy_Traffic_JunctionEnv, self).get_obs_agent(agent_id)
        if np.random.rand() <= self.failure_obs_prob:
            return -1.0 * obs
        return obs

    def reset(self):
        """
        Reset the environment and perform random initial steps.
        Returns:
          obs_list: zeroed observations to avoid initial leakage
          state: full joint state
        """
        # Initial reset
        obs_list, state = super(Messy_Traffic_JunctionEnv, self).reset()
        terminated = False
        steps = 0
        # Randomly step through env to scramble
        while steps < self.init_random_steps:
            actions = []
            for agent_id in range(self.nagents):
                avail = self.get_avail_agent_actions(agent_id)
                idx = np.nonzero(avail)[0]
                actions.append(np.random.choice(idx))
            _, terminated, _ = self.step(actions)
            steps += 1
            if terminated:
                obs_list, state = super(Messy_Traffic_JunctionEnv, self).reset()
        # Return zeroed observations to hide initial state
        zero_obs = [np.zeros_like(o) for o in obs_list]
        return zero_obs, state
