

import os
import logging
from .inference_service import JointLKInferenceService
from .data_transformer import GraphDataTransformer
from modeling.modeling_jointlk import DataLoader
# --- 配置 ---

MODEL_CHECKPOINT_PATH = r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\saved_models\medqa_usmle\medqa_usmle.model.pt.dev_38.0-test_39.8"
CONCEPT_EMB_PATH = r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\data\cpnet\concept.txt"  # 概念嵌入路径（来自仓库data/cpnet）
BASE_MODEL_NAME = 'roberta-large'
MAX_RELATIONS = 38  # MedQA的关系数（参考仓库训练脚本）
GNN_LAYERS = 5  # 与训练一致（参考sbatch脚本）
GNN_DIM = 200  # 与训练一致


class IntegrationFacade:
    """
    集成外观模式，统一管理 JointLK 服务和数据转换器。
    """
    def __init__(self):
        self.inference_service = JointLKInferenceService(
            model_checkpoint_path=MODEL_CHECKPOINT_PATH,
            concept_emb_path=CONCEPT_EMB_PATH,  # 新增
            model_name=BASE_MODEL_NAME,
            num_relations=MAX_RELATIONS,
            k=GNN_LAYERS,  # 新增
            gnn_dim=GNN_DIM  # 新增
        )
        self.transformer = None

    def initialize(self):
        """加载模型并初始化转换器。"""
        try:
            self.inference_service.load_model()
            self.transformer = GraphDataTransformer(
                cpnet_vocab_path=CONCEPT_EMB_PATH,
                tokenizer=self.inference_service.tokenizer,
                max_seq_len=128  # 可配置
            )
            logging.info("JointLK IntegrationFacade initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize IntegrationFacade: {e}")
            raise

    def process_query(self, question: str, nodes: list, relationships: list, detail: bool = False) -> dict:
        """
        完整的处理流程：转换数据 -> 执行推理。
        """
        if not self.transformer:
            return {"error": "Service not initialized."}

        try:
            #  数据转换

            # prepared_data = self.transformer.format_for_jointlk(nodes, relationships, question)
            prepared_data = DataLoader(self.inference_service.args,  r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\data\csqa\statement\dev.statement.jsonl", r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\data\csqa\graph\dev.graph.adj.pk",
                                          r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\data\csqa\statement\dev.statement.jsonl", r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\data\csqa\graph\dev.graph.adj.pk",
                                          r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\data\csqa\statement\dev.statement.jsonl", r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\data\csqa\graph\dev.graph.adj.pk",
                                          batch_size=10, eval_batch_size=10,
                                          device=(self.inference_service.device, self.inference_service.device),
                                          model_name=self.inference_service.args.encoder,
                                          max_node_num=self.inference_service.args.max_node_num, max_seq_length=self.inference_service.args.max_seq_len,
                                          is_inhouse=False, inhouse_train_qids_path=False,
                                          subsample=1.0, use_cache=True)
            print("prepared_data", prepared_data)

            prepared_data = self.inference_service.model(prepared_data, detail=True)
            print("到这里")
            #  模型推理
            inference_result = self.inference_service.run_inference(
                prepared_data,
                detail=detail # Pass flag here
            )

            # 组装返回结果
            return {
                "answer": inference_result.get("answer"),
                "visualization_data": inference_result.get("visualization_data"), # Propagate data
                "source": "JointLK Enhanced Reasoning",
                "metadata": prepared_data.get("metadata") # Include metadata if available/needed
            }
        except Exception as e:
            logging.error(f"Error processing query through JointLK facade: {e}", exc_info=True)
            return {"error": "An internal error occurred during reasoning."}


# --- 单例模式 ---
# 创建一个全局实例，以便在 main.py 中调用
try:
    jointlk_facade = IntegrationFacade()
    jointlk_facade.initialize()
except Exception as e:
    print(e)
    jointlk_facade = None
    logging.critical("Failed to create and initialize JointLK Facade. JointLK features will be disabled.")