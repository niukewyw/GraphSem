from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .masia_learner import MASIALearner
from .masia_qplex_learner import MASIALearner as MASIAQPLEXLearner
from .roco_learner import ROCOLearner
from .GraphSem_learner import GraphSemLearner
from .GraphSem_no_comm_learner import GraphSem_no_comm_Learner
from .GraphSem_uniform_learner import GraphSem_uniform_Learner
from .GraphSem_no_gumbel_learner import GraphSem_no_gumbel_Learner
from .GraphSem_no_enc_learner import GraphSem_no_enc_Learner
from .GraphSem_no_weight_learner import GraphSem_no_weight_Learner
from .GraphSem_no_dec_learner import GraphSem_no_dec_Learner
from .GraphSem_no_graph_learner import GraphSem_no_graph_Learner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["masia_learner"] = MASIALearner
REGISTRY["masia_qplex_learner"] = MASIAQPLEXLearner
REGISTRY["roco_learner"] = ROCOLearner
REGISTRY["GraphSem_learner"] = GraphSemLearner
REGISTRY["GraphSem_no_comm_learner"] = GraphSem_no_comm_Learner
REGISTRY["GraphSem_uniform_learner"] = GraphSem_uniform_Learner
REGISTRY["GraphSem_no_gumbel_learner"] = GraphSem_no_gumbel_Learner
REGISTRY["GraphSem_no_enc_learner"] = GraphSem_no_enc_Learner
REGISTRY["GraphSem_no_weight_learner"] = GraphSem_no_weight_Learner
REGISTRY["GraphSem_no_dec_learner"] = GraphSem_no_dec_Learner
REGISTRY["GraphSem_no_graph_learner"] = GraphSem_no_graph_Learner