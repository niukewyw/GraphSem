import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSem_no_comm_agent(nn.Module):
    def __init__(self, input_shape, args):
        super(GraphSem_no_comm_agent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(args.hidden_dim, args.n_actions))

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h_out = self.rnn(x, h_in)

        # No communication - directly use h_out for Q-value computation
        q = self.fc3(h_out)

        # Return dummy weights for compatibility
        # Reshape to match the expected format: [batch_size, n_agents, n_agents]
        dummy_weights = torch.zeros(h_out.shape[0] // self.args.n_agents, self.args.n_agents, self.args.n_agents, device=h_out.device)
        for i in range(self.args.n_agents):
            dummy_weights[:, i, i] = 1.0  # Only self-connection

        return q, h_out, dummy_weights 