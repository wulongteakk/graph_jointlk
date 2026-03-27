import tempfile

from src.jointlk_integration.data_transformer import GraphDataTransformer


def _dummy_tokenizer(text, max_length, padding, truncation, return_tensors):
    import torch
    return {
        "input_ids": torch.ones((1, max_length), dtype=torch.long),
        "attention_mask": torch.ones((1, max_length), dtype=torch.long),
    }


def test_transformer_accepts_graph_query_style_payload():
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as vocab:
        vocab.write("worker\nfall\n")
        vocab.flush()
        transformer = GraphDataTransformer(
            tokenizer=_dummy_tokenizer,
            cpnet_vocab_path=vocab.name,
            max_seq_len=16,
            max_node_num=8,
        )

        nodes = [
            {"element_id": "4:abc", "properties": {"name": "worker"}},
            {"element_id": "4:def", "properties": {"name": "fall"}},
        ]
        relationships = [
            {
                "type": "related_to",
                "start_node_element_id": "4:abc",
                "end_node_element_id": "4:def",
            }
        ]
        result = transformer.format_for_jointlk(nodes, relationships, "what happened?")

        assert result["metadata"]["processed_nodes"] == 2
        assert result["metadata"]["processed_edges"] == 2
        assert tuple(result["adj"][0].shape) == (2, 2)