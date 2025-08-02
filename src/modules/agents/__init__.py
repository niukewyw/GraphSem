REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .full_comm_agent import FullCommAgent
from .masia_agent import MASIAAgent
from .roco_agent import ROCOAgent
from .GraphSem_agent import GraphSem_agent
from .GraphSem_no_comm_agent import GraphSem_no_comm_agent
from .GraphSem_uniform_agent import GraphSem_uniform_agent
from .GraphSem_no_gumbel_agent import GraphSem_no_gumbel_agent
from .GraphSem_no_enc_agent import GraphSem_no_enc_agent
from .GraphSem_no_weight_agent import GraphSem_no_weight_agent
from .GraphSem_no_dec_agent import GraphSem_no_dec_agent
from .GraphSem_no_graph_agent import GraphSem_no_graph_agent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["full_comm"] = FullCommAgent
REGISTRY["masia"] = MASIAAgent
REGISTRY["roco"] = ROCOAgent
REGISTRY["GraphSem"] = GraphSem_agent
REGISTRY["GraphSem_no_comm"] = GraphSem_no_comm_agent
REGISTRY["GraphSem_uniform"] = GraphSem_uniform_agent
REGISTRY["GraphSem_no_gumbel"] = GraphSem_no_gumbel_agent
REGISTRY["GraphSem_no_enc"] = GraphSem_no_enc_agent
REGISTRY["GraphSem_no_weight"] = GraphSem_no_weight_agent
REGISTRY["GraphSem_no_dec"] = GraphSem_no_dec_agent
REGISTRY["GraphSem_no_graph"] = GraphSem_no_graph_agent