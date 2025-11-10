import networkx as nx
import os
from typing import List, Dict, Any
from src.logger import logger

EXPORT_DIR = "asset"
CONCEPT_FILE = os.path.join(EXPORT_DIR, "my_concept.txt")
GPICKLE_FILE = os.path.join(EXPORT_DIR, "my_graph.gpickle")


def generate_gpickle_export(query_nodes: List[Dict[str, Any]], query_relations: List[Dict[str, Any]]):
    """
    生成 concept.txt 和 graph.gpickle.


    query_nodes: 从 export_concept() 返回的节点列表。
                 格式: [ {'name': 'Node1'}, {'name': 'Node2'} ]
    query_relations: 从 export_concept() 返回的关系列表。
                     格式: [ {'head': 'Node1', 'tail': 'Node2', 'rel_type': 'RELATES_TO'} ]
    """

    try:
        # 确保导出目录存在
        os.makedirs(EXPORT_DIR, exist_ok=True)

        #  导出 concept.txt (节点词汇表) ---
        logger.info(f"Exporting nodes to {CONCEPT_FILE}...")


        id2concept = [record["name"] for record in query_nodes if record.get("name")]

        # 去重并排序，确保每次导出的 ID 映射一致
        id2concept = sorted(list(set(id2concept)))

        # 保存 concept.txt (用于 Grounding)
        with open(CONCEPT_FILE, "w", encoding="utf-8") as f:
            for concept in id2concept:
                f.write(f"{concept}\n")

        # 创建 concept_to_id 映射
        concept2id = {name: i for i, name in enumerate(id2concept)}
        logger.info(f"Exported {len(id2concept)} unique concepts.")

        # 动态创建 relation_to_id 映射 ---
        # 从查询结果中动态获取所有唯一的关系类型
        all_rel_types = sorted(list(set(record["rel_type"] for record in query_relations if record.get("rel_type"))))
        relation_to_id = {name: i for i, name in enumerate(all_rel_types)}
        N_REL = len(relation_to_id)

        logger.info(f"Found {N_REL} unique relationship types.")
        logger.info(f"Relation mapping (relation_to_id): {relation_to_id}")
        logger.info(f"Total relations for JointLK (N_REL * 2): {N_REL * 2}")

        # 导出 graph.gpickle (图结构) ---
        logger.info(f"Exporting graph structure to {GPICKLE_FILE}...")
        G = nx.MultiDiGraph()  # JointLK 使用 MultiDiGraph

        #  添加所有节点，使用它们的整数 ID
        for i in range(len(concept2id)):
            G.add_node(i)

        edges_added = 0
        #  遍历查询到的关系
        for record in query_relations:
            h_name, t_name, rel_type = record.get("head"), record.get("tail"), record.get("rel_type")

            # 确保头节点、尾节点和关系类型都有效
            if h_name not in concept2id or t_name not in concept2id or rel_type not in relation_to_id:
                continue

            h_id = concept2id[h_name]
            t_id = concept2id[t_name]
            rel_id = relation_to_id[rel_type]

            # 添加正向边和反向边
            G.add_edge(h_id, t_id, rel=rel_id, weight=1.0)
            # JointLK 期望反向边的 rel_id 是 (rel_id + N_REL)
            G.add_edge(t_id, h_id, rel=rel_id + N_REL, weight=1.0)
            edges_added += 2

        # 保存 .gpickle (用于 Graph Generation)
        nx.write_gpickle(G, GPICKLE_FILE)
        logger.info(f"Exported graph with {G.number_of_nodes()} nodes and {edges_added} edges (including reverse).")

    except Exception as e:
        logger.error(f"Error during graph export: {e}")
        raise e