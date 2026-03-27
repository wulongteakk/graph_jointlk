
import logging
import os

from .data_transformer import GraphDataTransformer
from .inference_service import JointLKInferenceService


MODEL_CHECKPOINT_PATH = os.getenv("JOINTLK_MODEL_CHECKPOINT", "")
CPNET_VOCAB_PATH = os.getenv("JOINTLK_CPNET_VOCAB", "")
CONCEPT_EMB_PATH = os.getenv("JOINTLK_CONCEPT_EMB", "")
BASE_MODEL_NAME = os.getenv("JOINTLK_BASE_MODEL", "roberta-large")
MAX_RELATIONS = int(os.getenv("JOINTLK_NUM_RELATIONS", "38"))
GNN_LAYERS = int(os.getenv("JOINTLK_GNN_LAYERS", "5"))
GNN_DIM = int(os.getenv("JOINTLK_GNN_DIM", "200"))
MAX_SEQ_LEN = int(os.getenv("JOINTLK_MAX_SEQ_LEN", "128"))
MAX_NODE_NUM = int(os.getenv("JOINTLK_MAX_NODE_NUM", "200"))


class IntegrationFacade:
    """统一管理 JointLK 初始化、输入转换和推理。"""

    def __init__(self):
        self.inference_service = JointLKInferenceService(
            model_checkpoint_path=MODEL_CHECKPOINT_PATH,
            concept_emb_path=CONCEPT_EMB_PATH,
            model_name=BASE_MODEL_NAME,
            num_relations=MAX_RELATIONS,
            k=GNN_LAYERS,
            gnn_dim=GNN_DIM,
        )
        self.transformer = None

    def initialize(self):
        if not CPNET_VOCAB_PATH:
            raise ValueError("JOINTLK_CPNET_VOCAB is required for GraphDataTransformer.")
        self.inference_service.load_model()
        self.transformer = GraphDataTransformer(
            cpnet_vocab_path=CPNET_VOCAB_PATH,
            tokenizer=self.inference_service.tokenizer,
            max_seq_len=MAX_SEQ_LEN,
            max_node_num=MAX_NODE_NUM,
        )
        logging.info("JointLK IntegrationFacade initialized successfully.")

    def process_query(self, question: str, nodes: list, relationships: list, detail: bool = False) -> dict:
        if not self.transformer:
            return {"error": "Service not initialized."}
        if not isinstance(nodes, list) or not isinstance(relationships, list):
            return {"error": "Invalid graph payload: nodes/relationships must be list."}

        try:
            prepared_data = self.transformer.format_for_jointlk(nodes, relationships, question)
            metadata = prepared_data.get("metadata", {})
            if metadata.get("processed_nodes", 0) <= 0:
                return {
                    "error": "No usable concept node can be mapped into JointLK vocabulary.",
                    "metadata": metadata,
                }
            inference_result = self.inference_service.run_inference(prepared_data, detail=detail)
            return {
                "answer": inference_result.get("answer"),
                "visualization_data": inference_result.get("visualization_data"),
                "source": "JointLK Enhanced Reasoning",
                "metadata": metadata,
            }
        except Exception as e:
            logging.error("Error processing query through JointLK facade: %s", e, exc_info=True)
            return {"error": "An internal error occurred during reasoning."}




try:
    jointlk_facade = IntegrationFacade()
    jointlk_facade.initialize()
except Exception:
    jointlk_facade = None
    logging.critical("Failed to create and initialize JointLK Facade. JointLK features will be disabled.")