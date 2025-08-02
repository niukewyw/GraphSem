REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .masia_controller import MASIAMAC
from .roco_controller import ROCOMAC
from .GraphSem_controller import GraphSemMAC
from .GraphSem_no_comm_controller import GraphSem_no_comm_MAC
from .GraphSem_uniform_controller import GraphSem_uniform_MAC
from .GraphSem_no_gumbel_controller import GraphSem_no_gumbel_MAC
from .GraphSem_no_enc_controller import GraphSem_no_enc_MAC
from .GraphSem_no_weight_controller import GraphSem_no_weight_MAC
from .GraphSem_no_dec_controller import GraphSem_no_dec_MAC
from .GraphSem_no_graph_controller import GraphSem_no_graph_MAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["masia_mac"] = MASIAMAC
REGISTRY["roco_mac"] = ROCOMAC
REGISTRY["GraphSem_mac"] = GraphSemMAC
REGISTRY["GraphSem_no_comm_mac"] = GraphSem_no_comm_MAC
REGISTRY["GraphSem_uniform_mac"] = GraphSem_uniform_MAC
REGISTRY["GraphSem_no_gumbel_mac"] = GraphSem_no_gumbel_MAC
REGISTRY["GraphSem_no_enc_mac"] = GraphSem_no_enc_MAC
REGISTRY["GraphSem_no_weight_mac"] = GraphSem_no_weight_MAC
REGISTRY["GraphSem_no_dec_mac"] = GraphSem_no_dec_MAC
REGISTRY["GraphSem_no_graph_mac"] = GraphSem_no_graph_MAC
