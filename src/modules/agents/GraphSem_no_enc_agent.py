import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers.transformer_decoder import TransformerDecoder


class GraphSem_no_enc_agent(nn.Module):
    def __init__(self, input_shape, args):
        super(GraphSem_no_enc_agent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim * args.n_agents, args.hidden_dim)
        self.fc3 = nn.Sequential(nn.Linear(args.hidden_dim * 2, args.hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(args.hidden_dim, args.n_actions))
        self.decoder = TransformerDecoder(args.hidden_dim, args.hidden_dim, args.hidden_dim,
                                          args.att_dec_dim, args.hidden_dim, args.hidden_dim,
                                          args.hidden_dim * 2, args.dec_att_heads, args.num_layers, args.dropout)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h_out = self.rnn(x, h_in)

        # communicate - w/o Enc: directly use historical encoded information
        h = h_out.clone()
        h = h.reshape(-1, self.args.n_agents, self.args.hidden_dim)
        comm = h.reshape(-1, 1, self.args.n_agents * self.args.hidden_dim).repeat(1, self.args.n_agents, 1)
        
        # Use uniform weights instead of hard attention (no encoder)
        # Create uniform weights matrix (1/n_agents for all connections)
        uniform_weights = torch.ones(h.shape[0], self.args.n_agents, self.args.n_agents, device=comm.device) / self.args.n_agents
        
        # Apply uniform weights to communication
        repeat_weights = torch.repeat_interleave(uniform_weights, repeats=self.args.hidden_dim, dim=-1)
        mask_comm = comm * repeat_weights
        enc_out = F.relu(self.fc2(mask_comm))

        # decoder
        dec_out = self.decoder(enc_out).reshape(-1, self.args.hidden_dim)

        # q
        q = torch.cat((h_out, dec_out), dim=-1)
        q = self.fc3(q)

        # Reshape uniform_weights to match the expected format: [batch_size, n_agents, n_agents]
        uniform_weights = uniform_weights.reshape(h.shape[0], self.args.n_agents, self.args.n_agents)
        return q, h_out, uniform_weights 