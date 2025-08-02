from smac.env import StarCraft2Env
from utils.dict2namedtuple import convert
import numpy as np

class MessyStarCraft(StarCraft2Env):
    def __init__(self, **kwargs):
        # Unpack arguments from sacred
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.init_random_steps = getattr(args, "randomize_initial_state", 5)
        self.failure_obs_prob = getattr(args, "failure_obs_prob", 0.10)
        args = args._asdict()
        args.pop("randomize_initial_state", 5)
        args.pop("failure_obs_prob", 0.10)
        super(MessyStarCraft, self).__init__(**args)

    def get_obs_agent(self, agent_id):
        default_obs = super(MessyStarCraft, self).get_obs_agent(agent_id)
        # if np.random.rand() <= self.failure_obs_prob:
        #     noise = np.random.normal(0, 0.1, size=default_obs.shape) 
        #     return np.clip(default_obs + noise, -1, 1)

        return default_obs

    def reset(self):
        super(MessyStarCraft, self).reset()
        terminated = False
        steps = 0
        nr_random_steps = self.init_random_steps
        while steps < nr_random_steps:
            actions = []
            for agent_id in range(self.n_agents):
                avail_actions = self.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)
            _, terminated, _ = self.step(actions)
            steps += 1
            if terminated:
                super(MessyStarCraft, self).reset()
        observations = self.get_obs()
        return [np.zeros_like(obs) for obs in observations], self.get_state()
    
    def get_obs_component(self):
        move_feats_dim = self.get_obs_move_feats_size()
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()
        obs_component = [move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim]
        # print(obs_component)
        return obs_component

    def get_state_component(self):
        if self.obs_instead_of_state:
            return [self.get_obs_size()] * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = [ally_state, enemy_state]

        if self.state_last_action:
            size.append(self.n_agents * self.n_actions)
        if self.state_timestep_number:
            size.append(1)
        return size

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "n_enemies": self.n_enemies,
            "episode_limit": self.episode_limit,

            "n_normal_actions": self.n_actions_no_attack,
            "n_allies": self.n_agents - 1,
            # "obs_ally_feats_size": self.get_obs_ally_feats_size(),
            # "obs_enemy_feats_size": self.get_obs_enemy_feats_size(),
            "state_ally_feats_size": self.get_ally_num_attributes(),  # 4 + self.shield_bits_ally + self.unit_type_bits,
            "state_enemy_feats_size": self.get_enemy_num_attributes(),
            # 3 + self.shield_bits_enemy + self.unit_type_bits,
            "obs_component": self.get_obs_component(),
            "state_component": self.get_state_component(),
            "map_type": self.map_type,
        }
        # print(env_info)
        return env_info

    def _get_medivac_ids(self):
        medivac_ids = []
        for al_id, al_unit in self.agents.items():
            if self.map_type == "MMM" and al_unit.unit_type == self.medivac_id:
                medivac_ids.append(al_id)
        print(medivac_ids)  # [9]
        return medivac_ids
