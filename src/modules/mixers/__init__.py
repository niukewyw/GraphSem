REGISTRY = {}

from .GraphSem_qmix import GraphSem_qmix
from .GraphSem_mean_pool_qmix import GraphSem_mean_pool_qmix
from .GraphSem_no_comm_qmix import GraphSem_no_comm_qmix
from .GraphSem_no_enc_qmix import GraphSem_no_enc_qmix
from .GraphSem_no_weight_qmix import GraphSem_no_weight_qmix
from .GraphSem_no_dec_qmix import GraphSem_no_dec_qmix
from .GraphSem_no_graph_qmix import GraphSem_no_graph_qmix
from .qmix import QMixer

REGISTRY["GraphSem_qmix"] = GraphSem_qmix
REGISTRY["GraphSem_mean_pool_qmix"] = GraphSem_mean_pool_qmix
REGISTRY["GraphSem_no_comm_qmix"] = GraphSem_no_comm_qmix
REGISTRY["GraphSem_no_enc_qmix"] = GraphSem_no_enc_qmix
REGISTRY["GraphSem_no_weight_qmix"] = GraphSem_no_weight_qmix
REGISTRY["GraphSem_no_dec_qmix"] = GraphSem_no_dec_qmix
REGISTRY["GraphSem_no_graph_qmix"] = GraphSem_no_graph_qmix
REGISTRY["qmix"] = QMixer