from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, teacher_forcing=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if teacher_forcing:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, teacher_forcing=True)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        if teacher_forcing:
            log_prefix = "teacher_forcing_" + log_prefix
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
        
def tamper_obs(obs, prob, noise, obs_component):
    tampered_obs = []
    component_lengths = []
    offsets = [0]
    for comp in obs_component:
        if isinstance(comp, (int, float)):
            length = comp
        elif isinstance(comp, (tuple, list)):
            length = np.prod(comp)
        else:
            raise ValueError(f"Unknown component type in obs_component: {comp}")
        component_lengths.append(length)
        offsets.append(offsets[-1] + length)
    offsets.pop()
    
    for arr in obs:
        if np.random.rand() < prob:
            is_all_zeros = all(x == 0 for x in arr)
            if is_all_zeros:
                tampered_obs.append(arr)
            else:
                tampered_arr = arr.copy()
                parts = []
                for i, length in enumerate(component_lengths):
                    start = offsets[i]
                    end = start + length
                    parts.append(tampered_arr[start:end])

                for i, part in enumerate(parts):
                    comp = obs_component[i]
                    if isinstance(comp, int):
                        if i == 0:  # move_feats
                            for j in range(len(part)):
                                if np.random.rand() < noise:
                                    part[j] = 1 - part[j] if part[j] in [0, 1] else part[j]
                        elif i == len(parts) - 1:  # own_feats
                            gaussian_noise = np.random.normal(loc=0.0, scale=noise)
                            part[0] = np.clip(part[0] + gaussian_noise, 0, 1)
                            current_one_idx = None
                            for idx in range(1, len(part)):
                                if part[idx] == 1.0:
                                    current_one_idx = idx - 1
                                    break
                            possible_indices = list(range(len(part) - 1))
                            if current_one_idx is not None and len(possible_indices) > 1:
                                possible_indices.remove(current_one_idx)
                            new_one_idx = np.random.choice(possible_indices)
                            part[1:] = 0.0
                            part[1 + new_one_idx] = 1.0
                    else:  # enemy_feats, ally_feats
                        group_size = comp[-1] if isinstance(comp, (tuple, list)) else 8
                        for j in range(0, len(part), group_size):
                            chunk = part[j:j + group_size]
                            if len(chunk) == group_size:
                                chunk[0] = 1.0 if chunk[0] == 0.0 else 0.0
                                for k in range(1, min(5, group_size)):
                                    gaussian_noise = np.random.normal(loc=0.0, scale=noise)
                                    chunk[k] = np.clip(chunk[k] + gaussian_noise, -1, 1)
                                if group_size > 5:
                                    current_one_idx = None
                                    for idx in range(5, group_size):
                                        if chunk[idx] == 1.0:
                                            current_one_idx = idx - 5
                                            break
                                    possible_indices = list(range(group_size - 5))
                                    if current_one_idx is not None and len(possible_indices) > 1:
                                        possible_indices.remove(current_one_idx)
                                    new_one_idx = np.random.choice(possible_indices)
                                    chunk[5:] = 0.0
                                    chunk[5 + new_one_idx] = 1.0
                            part[j:j + group_size] = chunk
                
                tampered_arr = np.concatenate(parts)
                tampered_obs.append(tampered_arr)
        else:
            tampered_obs.append(arr)
    
    return tampered_obs


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data[0]
            OBS_component = data[1]
            OBStamper_prob = data[2][0]
            OBStamper_noise = data[2][1]
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            obs_real = obs
            obs = tamper_obs(obs, OBStamper_prob, OBStamper_noise, OBS_component)
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "obs_real": obs_real,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs(),
                "obs_real": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "save_replay":
            remote.send(env.save_replay())
        else:
            raise NotImplementedError