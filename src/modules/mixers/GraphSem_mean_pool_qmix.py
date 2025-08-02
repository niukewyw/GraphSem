import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from modules.layers.utils import normalization


class GraphSem_mean_pool_qmix(nn.Module):
    def __init__(self, args):
        super(GraphSem_mean_pool_qmix, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_dim = int(np.prod(args.obs_shape))
        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.input_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.input_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.input_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.input_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        self.hyper_b_1 = nn.Linear(self.input_dim, self.embed_dim)
        self.V = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, obs, adjacency):
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        # Use mean pooling instead of graph attention pooling
        node_features = obs.reshape(-1, self.n_agents, self.input_dim)
        # Simple mean pooling across agents
        pool_state = torch.mean(node_features, dim=1)  # [batch_size, input_dim]

        w1 = torch.abs(self.hyper_w_1(pool_state))
        b1 = self.hyper_b_1(pool_state)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = torch.abs(self.hyper_w_final(pool_state))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # State-dependent bias
        v = self.V(pool_state).view(-1, 1, 1)

        # Compute final output
        y = torch.bmm(hidden, w_final) + v

        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot 