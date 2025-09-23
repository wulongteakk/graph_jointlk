#
# 文件名: backend/src/qagnn_scorer.py (已修正)
# 描述: 从 'qagnn' (仓库2) 移植过来的节点评分和剪枝模块。
#

import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import RobertaTokenizer, RobertaForMaskedLM, logging as hf_logging
import logging
from typing import Set, Tuple, List, Dict, Any

# 设置日志
log = logging.getLogger(__name__)

# 抑制 transformers 的一些警告信息
hf_logging.set_verbosity_error()


# --- 1. 从 qagnn/modeling/modeling_qagnn.py 复制的核心类 ---
class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    """
    这是 'qagnn' (仓库2) 中用于计算节点得分的自定义 RoBERTa 模型。
    它通过计算 "问题 + 节点" 句子的 Masked LM 损失来评估相关性。
    损失越低 = 相关性越高。
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                masked_lm_labels=None):
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                                      masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
        return outputs


# --- 2. 全局加载评分模型 (只在启动时加载一次) ---
TOKENIZER = None
LM_MODEL = None
DEVICE = None
QAGNN_SCORER_ENABLED = False

try:
    log.info('Loading QAGNN LM scorer (roberta-large)...')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
    LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
    LM_MODEL.to(DEVICE)
    LM_MODEL.eval()
    log.info(f'QAGNN LM scorer loaded successfully onto {DEVICE}.')
    QAGNN_SCORER_ENABLED = True
except Exception as e:
    log.error(f"************************************************************")
    log.error(f"Error loading QAGNN LM scorer (roberta-large): {e}")
    log.error("The dynamic pruning module will be DISABLED.")
    log.error("Please check torch, transformers, and CUDA setup.")
    log.error(f"************************************************************")
    QAGNN_SCORER_ENABLED = False


# --- 3. 移植和修改 get_LM_score 函数 ---
def get_LM_score_for_nodes(node_names: List[str], question: str) -> 'OrderedDict[str, float]':
    """
    为一批节点名称（字符串）和
    一个问题（字符串）计算相关性得分。
    """
    if LM_MODEL is None or TOKENIZER is None:
        raise ImportError("QAGNN Scorer models (Roberta) are not loaded.")

    sents, scores = [], []
    # QA-GNN 的逻辑是：上下文节点就是问题本身
    sents.append(TOKENIZER.encode(question.lower(), add_special_tokens=True, max_length=512, truncation=True))

    # 为每个节点创建 "问题 + 节点" 的句子
    for node_name in node_names:
        # 原始格式是 "question. concept."
        sent = '{} {}.'.format(question.lower(), str(node_name).replace("_", " "))
        sent_encoded = TOKENIZER.encode(sent, add_special_tokens=True, max_length=512, truncation=True)
        sents.append(sent_encoded)

    n_nodes = len(sents)  # n_nodes = 1 (问题) + len(node_names)
    cur_idx = 0
    batch_size = 32  # 可以根据您的VRAM调整

    with torch.no_grad():
        while cur_idx < n_nodes:
            # 准备批次
            input_ids = sents[cur_idx: cur_idx + batch_size]
            max_len = max([len(seq) for seq in input_ids])

            padded_input_ids = []
            for seq in input_ids:
                padded_seq = seq + [TOKENIZER.pad_token_id] * (max_len - len(seq))
                padded_input_ids.append(padded_seq)

            input_ids_tensor = torch.tensor(padded_input_ids).to(DEVICE)
            mask = (input_ids_tensor != TOKENIZER.pad_token_id).long()

            # 运行模型 (QA-GNN的逻辑是计算MLM损失)
            outputs = LM_MODEL(input_ids_tensor, attention_mask=mask, masked_lm_labels=input_ids_tensor)
            loss = outputs[0]  # [B, ]

            # 得分是负损失（损失越低，相关性越高）
            _scores = list(-loss.detach().cpu().numpy())
            scores += _scores
            cur_idx += batch_size

    assert len(scores) == n_nodes

    # 我们将返回一个 {node_name: score} 的字典
    node_scores = scores[1:]
    node_names_with_scores = list(zip(node_names, node_scores))

    # 按得分从高到低排序
    sorted_nodes = sorted(node_names_with_scores, key=lambda x: -x[1])

    return OrderedDict(sorted_nodes)


# --- 4. 我们的主包装函数，将用于 "仓库1" (QA_integration_new.py) ---
def prune_and_score_nodes(
        nodes_list: List[Dict[str, Any]],
        question: str,
        top_k: int = 15
) -> Set[str]:
    """

    Args:
        nodes_list: 从 get_graph_response 提取的实体列表
                       e.g., [{'element_id': ..., 'labels': ['User'], 'properties': {'id': ''}}, ...]
        question: 用户的原始问题.
        top_k: 保留得分最高的 k 个节点。
               设置为 0 或 负数 则不剪枝。

    Returns:
        Set[str]: 一个包含所有被保留节点名称的集合。
    """

    if not nodes_list or not QAGNN_SCORER_ENABLED:
        if not QAGNN_SCORER_ENABLED:
            log.warning("QAGNN Scorer is disabled, returning all original nodes.")
        # 返回所有节点名称
        all_node_names = set()
        for entity in nodes_list:
            props = entity.get('properties', {})
            # 修正：同时检查 'id' 和 'name'，并优先使用 'id'
            node_name = props.get('id', props.get('name'))
            if node_name:
                all_node_names.add(node_name)
        return all_node_names

    log.info(f"[QAGNN Scorer] Pruning ndoes list with {len(nodes_list)} items...")

    # 1. 从 'entities' 列表中提取所有唯一的节点名称
    #    *** 这是关键的修正 ***
    all_nodes_names = set()
    for entity in nodes_list:
        props = entity.get('properties', {})
        # 修正：同时检查 'id' 和 'name'，并优先使用 'id'
        # 根据您的日志, 属性键是 'id'
        node_name = props.get('id', props.get('name'))

        # 针对 Document 节点的特殊处理
        if not node_name and 'Document' in entity.get('labels', []):
            node_name = props.get('fileName')

        if node_name:
            all_nodes_names.add(node_name)

    if not all_nodes_names:
        log.warning("[QAGNN Scorer] No nodes with 'id' or 'name' property found in entities list. Returning all.")
        return set()  # 返回空集合，后续逻辑会处理

    node_names_list = list(all_nodes_names)
    log.debug(f"[QAGNN Scorer] Found {len(node_names_list)} unique nodes to score.")

    # 2. 调用 QAGNN 评分器
    try:
        node_scores: 'OrderedDict[str, float]' = get_LM_score_for_nodes(node_names_list, question)
    except Exception as e:
        log.error(f"[QAGNN Scorer] Error during node scoring: {e}")
        return all_nodes_names  # 评分失败，返回所有节点

    # 3. 确定要保留的节点
    retained_node_names_set: Set[str]
    if top_k > 0 and len(node_scores) > top_k:
        retained_node_names_set = set(list(node_scores.keys())[:top_k])
        log.info(f"[QAGNN Scorer] Retaining top {top_k} nodes out of {len(node_scores)}.")
        log.debug(f"[QAGNN Scorer] Top nodes: {retained_node_names_set}")
    else:
        retained_node_names_set = set(node_scores.keys())
        log.info(f"[QAGNN Scorer] Retaining all {len(retained_node_names_set)} scored nodes (top_k <= 0).")

    return retained_node_names_set