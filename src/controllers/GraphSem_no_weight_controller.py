import torch as th
from components.episode_buffer import EpisodeBatch
from modules.agents import REGISTRY as agent_REGISTRY
from modules.agents import GraphSem_no_weight_agent
from components.action_selectors import REGISTRY as action_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from components.standarize_stream import RunningMeanStd


class GraphSem_no_weight_MAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail = ep_batch["avail_actions"][:, t_ep] if t_ep is not None else ep_batch["avail_actions"]
        agent_outputs = self.forward(ep_batch, t=t_ep)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t] if t is not None else ep_batch["avail_actions"]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their impact on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size, self.n_agents, -1)
                agent_outs[avail_actions == 0.0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = GraphSem_no_weight_agent(self.args, input_shape, self.n_agents)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to agents
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape 