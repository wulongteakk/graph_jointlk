

import torch
import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Callable
from transformers import RobertaTokenizer, RobertaConfig
import numpy as np
import argparse
# --- 动态添加 JointLK 库到 Python 路径 ---

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../../'))  # 回退三层到 llm-graph-builder-private/
jointlk_path = os.path.join(project_root, 'JointLK')
print(f"jointlk_path: {jointlk_path}")  # 应输出 .../llm-graph-builder/JointLK
if jointlk_path not in sys.path:
    sys.path.insert(0, jointlk_path)  # 优先导入正确的 JointLK 目录


# --- 导入 JointLK 模型 ---
try:
    from modeling.modeling_jointlk import JOINT_LM_KG
except ImportError as e:
    logging.error(f"Failed to import JointLK model from {jointlk_path}. Error: {e}")

    #JointLK模型加载不成功也不出错

    class MockJointLKModel(torch.nn.Module):
        def __init__(self, config, *args, **kwargs):
            super().__init__()
            logging.warning("[MockJointLKModel] Using mock model class.")
            self.config = config

        def forward(self, input_ids, attention_mask, adj=None, **kwargs):
            logging.info(f"[MockJointLKModel] Mock inference call with input shape {input_ids.shape}.")

            return torch.rand(input_ids.shape[0], 5)


    JOINT_LM_KG = MockJointLKModel


class JointLKInferenceService:
    """
    封装 JointLK 模型的加载和推理过程。
    """

    def __init__(self, model_checkpoint_path: str,
                 concept_emb_path: str,  # 新增：概念嵌入路径
                 model_name: str = 'roberta-large',
                 num_relations: int = 38,
                 k: int = 5,  # GNN层数（与训练一致）
                 gnn_dim: int = 200,  # GNN维度（与训练一致）
                 att_head_num: int = 2,  # 注意力头数
                 fc_dim: int = 200,  # 全连接层维度
                 fc_layer_num: int = 2):  # 全连接层数

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_checkpoint_path = model_checkpoint_path
        self.concept_emb_path = concept_emb_path  # 概念嵌入路径
        self.model_name = model_name
        # 超参数（必须与预训练模型一致，参考训练脚本）
        self.encoder = model_name
        self.num_relations = num_relations
        self.k = k
        self.gnn_dim = gnn_dim
        self.att_head_num = att_head_num
        self.fc_dim = fc_dim
        self.fc_layer_num = fc_layer_num
        self.model = None
        self.tokenizer = None
        self.concept_emb = None  # 概念嵌入张量


    def load_model(self):
        """加载模型和分词器到内存。"""
        logging.info(f"Loading JointLK model. Base model: {self.model_name}, Device: {self.device}")
        try:
            # 加载分词器
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder)

            # 加载概念嵌入（必须的初始化参数）
            my_embedding_paths =[r"D:\NEO4J_related\llm-builder\llm\llm-graph-builder\JointLK\data\cpnet\tzw.ent.npy"]
            self.concept_emb =  [np.load(path) for path in my_embedding_paths]
            self.concept_emb = torch.tensor(np.concatenate(self.concept_emb, 1), dtype=torch.float)
            self.concept_num, self.concept_in_dim = self.concept_emb.shape
            target_num, target_dim = 9958, 768
            flat_emb = self.concept_emb.flatten()
            self.concept_emb = flat_emb[:target_num * target_dim].view(target_num, target_dim)
            self.concept_num, self.concept_in_dim = self.concept_emb.shape
            # 加载预训练权重
            if os.path.exists(self.model_checkpoint_path):
                logging.info(f"Loading weights from checkpoint: {self.model_checkpoint_path}")
                #self.model.load_state_dict(torch.load(self.model_checkpoint_path, map_location=self.device, weights_only=True))
                # 注册 argparse.Namespace 为安全全局
                torch.serialization.add_safe_globals([argparse.Namespace])

                # 然后正常加载模型
                model_state_dict, old_args = torch.load(
                    self.model_checkpoint_path,
                    map_location=self.device
                )
                self.args = old_args

                # 实例化模型
                self.model = JOINT_LM_KG(old_args, old_args.encoder, k=old_args.k, n_ntype=4, n_etype=old_args.num_relation, n_concept=self.concept_num,
                               concept_dim=old_args.gnn_dim,
                               concept_in_dim=self.concept_in_dim ,
                               n_attention_head=old_args.att_head_num, fc_dim=old_args.fc_dim, n_fc_layer=old_args.fc_layer_num,
                               p_emb=old_args.dropouti, p_gnn=old_args.dropoutg, p_fc=old_args.dropoutf,
                               pretrained_concept_emb=self.concept_emb, freeze_ent_emb=old_args.freeze_ent_emb,
                               init_range=old_args.init_range,
                               encoder_config={}
                )
                # print(checkpoint)
                self.model.load_state_dict(model_state_dict, strict=False)

            else:
                logging.warning(f"Checkpoint file not found at {self.model_checkpoint_path}. Using base model weights.")

            # 部署到设备并设为推理模式
            self.model.encoder.to(self.device)
            self.model.decoder.to(self.device)
            self.model.eval()

            logging.info("JointLK model loaded successfully.")

        except Exception as e:
            logging.error(f"Failed to load JointLK model: {e}", exc_info=True)
            raise

    def run_inference(self, prepared_input: Dict[str, Any],  detail: bool = False) -> str:
        if self.model is None or self.tokenizer is None:
            logging.error("Model not loaded. Call load_model() first.")
            return {"answer": "Error: Model not initialized.", "visualization_data": None}

        try:
            # 1. 提取输入并移动到设备
            input_ids = prepared_input["input_ids"].to(self.device)
            attention_mask = prepared_input["attention_mask"].to(self.device)
            concept_ids = prepared_input["concept_ids"].to(self.device)
            node_type_ids = prepared_input["node_type_ids"].to(self.device)
            node_scores = prepared_input["node_scores"].to(self.device)
            adj_lengths = prepared_input["adj_lengths"].to(self.device)
            edge_index, edge_type = prepared_input["adj"]
            edge_index = edge_index.to(self.device)
            edge_type = edge_type.to(self.device)

            # 取编码器输出（sent_vecs和last_hidden_states）
            # 编码器会输出句子向量和隐藏状态
            with torch.no_grad():
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                if hasattr(encoder_outputs, 'last_hidden_state'):
                    last_hidden_states = encoder_outputs.last_hidden_state
                else:
                    last_hidden_states = encoder_outputs[0]  # Fallback for tuple output
                sent_vecs = last_hidden_states[:, 0, :]  # 取[CLS]作为句子向量 (batch, hidden_dim)

            # 3. 调用模型forward（匹配modeling_jointlk.py的参数）
            with torch.no_grad():
                decoder_outputs = self.model.decoder(
                    sent_vecs=sent_vecs,
                    concept_ids=concept_ids,
                    node_type_ids=node_type_ids,
                    node_scores=node_scores,
                    adj_lengths=adj_lengths,
                    adj=(edge_index, edge_type),
                    last_hidden_states=last_hidden_states,
                    attention_mask=attention_mask,
                    detail=detail # Pass the detail flag here
                )
                logits = decoder_outputs[0]
                visualization_data = None
                if detail:
                    # In detail mode, outputs[2] contains visualization_data wrapped in another dict
                    # From JOINT_LM_KG.forward: return logits, attn, detailed_output -> detailed_output = outputs[2]
                    # From JointLK.forward: return logits, pool_attn, viz_data -> viz_data = outputs[2]
                    # We need to be precise about which return signature we are matching.
                    # Assuming self.model.decoder maps to JointLK.forward (return logits, pool_attn, viz_data)
                    if len(decoder_outputs) > 2:
                         # Extract visualization data from the third element returned by JointLK.forward
                         viz_data_from_model = decoder_outputs[2]
                         # If JOINT_LM_KG wraps this further, adjust extraction here.
                         # Based on previous modeling_jointlk.py change in JOINT_LM_KG.forward:
                         # detailed_output = {"visualization_data": viz_data} and return was detailed_output.
                         # Let's assume the most direct path: viz_data_from_model is the final visualization dictionary.
                         visualization_data = viz_data_from_model

            # 4. 解析结果（假设是分类任务，取最大概率索引）
            prediction_index = torch.argmax(logits, dim=-1).item()
            result_str = f"JointLK enhanced answer (Prediction Index: {prediction_index})"
            logging.info(f"Inference complete. Result: {result_str}")

            return {
                "answer": result_str,
                "visualization_data": visualization_data
            }

        except Exception as e:
            logging.error(f"Error during JointLK inference: {e}", exc_info=True)
            return {"answer": "Error during model reasoning.", "visualization_data": None}