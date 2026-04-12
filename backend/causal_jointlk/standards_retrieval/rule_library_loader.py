from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import RetrievalDoc
from .tokenizer import join_non_empty


class RuleLibraryLoader:
    def __init__(self, rule_library_path: str | Path):
        self.path = Path(rule_library_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Rule library not found: {self.path}")

    def load_raw(self) -> Dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_docs(self) -> tuple[list[RetrievalDoc], dict[str, Any]]:
        data = self.load_raw()
        docs: List[RetrievalDoc] = []
        docs.extend(self._load_clause_rules(data))
        docs.extend(self._load_basic_type_items(data))
        docs.extend(self._load_severity_items(data))
        docs.extend(self._load_industry_rules_and_items(data))
        docs.extend(self._load_coding_rules(data))
        return docs, data.get("retrieval_profiles", {})

    @staticmethod
    def _effective_from(data: Dict[str, Any], standard_no: str, fallback: Optional[str] = None) -> Optional[str]:
        meta = data.get("standard_meta", {})
        linked = meta.get("linked_standards", [])
        primary = meta.get("primary_standard", {})
        if primary and primary.get("standard_no") == standard_no:
            return primary.get("effective_from", fallback)
        for item in linked:
            if item.get("standard_no") == standard_no:
                return item.get("effective_from", fallback)
        return fallback

    def _load_clause_rules(self, data: Dict[str, Any]) -> List[RetrievalDoc]:
        out: List[RetrievalDoc] = []
        for rule in data.get("条款规则索引", []):
            dimension = rule.get("dimension", "unknown")
            if dimension == "industry":
                dimension = "industry_type"
            text = join_non_empty(
                [
                    rule.get("rule_text", ""),
                    f"触发条件: {json.dumps(rule.get('trigger_condition', {}), ensure_ascii=False)}",
                    f"裁决策略: {', '.join(rule.get('decision_policy', []))}",
                    f"证据要求: {', '.join(rule.get('evidence_requirements', []))}",
                    f"排除规则: {', '.join(rule.get('exclusion_rules', []))}",
                    f"编码映射: {json.dumps(rule.get('code_mapping', None), ensure_ascii=False)}",
                ]
            )
            out.append(
                RetrievalDoc(
                    doc_id=rule["rule_id"],
                    library_name="standard_rule_library",
                    dimension=dimension,
                    source_type="clause_rule",
                    standard_no=rule.get("standard_no"),
                    effective_from=rule.get("effective_from"),
                    clause_id=rule.get("clause_id"),
                    title=rule.get("rule_name", rule["rule_id"]),
                    text=text,
                    keywords=rule.get("decision_policy", []) + rule.get("evidence_requirements", []),
                    metadata={
                        "trigger_condition": rule.get("trigger_condition", {}),
                        "code_mapping": rule.get("code_mapping", None),
                    },
                )
            )
        return out

    def _load_basic_type_items(self, data: Dict[str, Any]) -> List[RetrievalDoc]:
        table = data.get("基本事故类型表", {})
        eff = self._effective_from(data, "GB 6441-2025", "2026-07-01")
        out: List[RetrievalDoc] = []
        for item in table.get("items", []):
            text = join_non_empty(
                [
                    item.get("standard_description", ""),
                    f"正例触发模式: {'；'.join(item.get('positive_trigger_patterns', []))}",
                    f"证据要求: {'、'.join(item.get('evidence_requirements', []))}",
                    f"排除规则: {'；'.join(item.get('exclusion_rules', []))}",
                    f"常见混淆对: {'、'.join(item.get('common_confusion_pairs', []))}",
                ]
            )
            keywords = list(dict.fromkeys(item.get("derived_keywords", []) + item.get("common_confusion_pairs", [])))
            out.append(
                RetrievalDoc(
                    doc_id=f"basic_type::{item['type_code_suffix']}",
                    library_name="standard_rule_library",
                    dimension="basic_accident_type",
                    source_type="basic_type_item",
                    standard_no="GB 6441-2025",
                    effective_from=eff,
                    clause_id=item.get("clause_id"),
                    title=item.get("type_name", ""),
                    text=text,
                    keywords=keywords,
                    metadata=item,
                )
            )
        return out

    def _load_severity_items(self, data: Dict[str, Any]) -> List[RetrievalDoc]:
        table = data.get("伤害程度表", {})
        eff_6441 = self._effective_from(data, "GB 6441-2025", "2026-07-01")
        eff_15499 = self._effective_from(data, "GB 15499-2025", "2027-01-01")
        out: List[RetrievalDoc] = []

        for item in table.get("items", []):
            text = join_non_empty(
                [
                    item.get("standard_description", ""),
                    f"判定逻辑: {'；'.join(item.get('judgement_logic', []))}",
                    f"GB15499衔接: {json.dumps(item.get('gb15499_linkage', {}), ensure_ascii=False)}",
                ]
            )
            out.append(
                RetrievalDoc(
                    doc_id=f"severity::{item['type_code_suffix']}",
                    library_name="standard_rule_library",
                    dimension="injury_severity",
                    source_type="severity_item",
                    standard_no="GB 6441-2025",
                    effective_from=eff_6441,
                    clause_id=item.get("clause_id"),
                    title=item.get("severity_name", ""),
                    text=text,
                    keywords=item.get("judgement_logic", []),
                    metadata=item,
                )
            )

        integration = table.get("gb15499_integration", {})
        for rule in integration.get("core_rules", []):
            out.append(
                RetrievalDoc(
                    doc_id=rule["rule_id"],
                    library_name="standard_rule_library",
                    dimension="injury_lost_workdays",
                    source_type="gb15499_core_rule",
                    standard_no=integration.get("standard_no", "GB 15499-2025"),
                    effective_from=integration.get("effective_from", eff_15499),
                    clause_id=rule.get("clause_id"),
                    title=rule.get("rule_name", rule["rule_id"]),
                    text=join_non_empty(
                        [
                            rule.get("rule_text", ""),
                            f"裁决策略: {', '.join(rule.get('decision_policy', []))}",
                            f"证据要求: {', '.join(rule.get('evidence_requirements', []))}",
                        ]
                    ),
                    keywords=rule.get("decision_policy", []) + rule.get("evidence_requirements", []),
                    metadata=rule,
                )
            )

        for appendix in integration.get("appendix_a_tables", []):
            notes = appendix.get("notes", [])
            table_common = {
                "library_name": "standard_rule_library",
                "dimension": "injury_lost_workdays",
                "source_type": "gb15499_appendix_row",
                "standard_no": integration.get("standard_no", "GB 15499-2025"),
                "effective_from": integration.get("effective_from", eff_15499),
                "clause_id": appendix.get("table_id"),
            }
            for idx, row in enumerate(appendix.get("rows", []), start=1):
                lost_workdays = row.get("lost_workdays")
                title = f"{appendix.get('table_name', '')} - {row.get('lookup_text', '')}"
                text = join_non_empty(
                    [
                        f"部位域: {appendix.get('body_domain', '')}",
                        f"换算路径: {row.get('lookup_text', '')}",
                        f"损失工作日: {lost_workdays}",
                        f"备注: {'；'.join(notes)}",
                    ]
                )
                keywords = [appendix.get("body_domain", ""), row.get("lookup_text", ""), str(lost_workdays)]
                out.append(
                    RetrievalDoc(
                        doc_id=f"{appendix['table_id']}::{idx:04d}",
                        title=title,
                        text=text,
                        keywords=[k for k in keywords if k],
                        metadata={**appendix, **row},
                        **table_common,
                    )
                )
        return out

    def _load_industry_rules_and_items(self, data: Dict[str, Any]) -> List[RetrievalDoc]:
        table = data.get("行业类型表", {})
        out: List[RetrievalDoc] = []
        ext = table.get("external_standard", {})
        standard_no = ext.get("standard_no", "GB/T 4754-2017")
        eff = ext.get("effective_from", "2017-10-01")

        for rule in table.get("classification_rules", []):
            out.append(
                RetrievalDoc(
                    doc_id=rule["rule_id"],
                    library_name="standard_rule_library",
                    dimension="industry_type",
                    source_type="industry_rule",
                    standard_no=standard_no,
                    effective_from=eff,
                    clause_id=rule.get("clause_id"),
                    title=rule.get("rule_name", rule["rule_id"]),
                    text=join_non_empty(
                        [
                            rule.get("rule_text", ""),
                            f"裁决策略: {', '.join(rule.get('decision_policy', []))}",
                            f"证据要求: {', '.join(rule.get('evidence_requirements', []))}",
                        ]
                    ),
                    keywords=rule.get("decision_policy", []) + rule.get("evidence_requirements", []),
                    metadata=rule,
                )
            )

        for item in table.get("items", []):
            title = f"{item.get('industry_code_gbt4754', '')} {item.get('industry_name', '')}".strip()
            text = join_non_empty(
                [
                    item.get("standard_description", ""),
                    f"层级: {item.get('industry_level', '')}",
                    f"父级: {item.get('parent_code', '') or '无'}",
                    f"层级路径: {' > '.join(item.get('hierarchy_path', []))}",
                    f"GB6441后四位: {item.get('gb6441_code_suffix', '')}",
                    f"同义称谓: {'、'.join(item.get('synonyms', []))}",
                    f"项目场景别名: {'、'.join(item.get('project_scene_aliases', []))}",
                ]
            )
            keywords = list(
                dict.fromkeys(
                    [item.get("industry_name", "")]
                    + item.get("synonyms", [])
                    + item.get("project_scene_aliases", [])
                    + item.get("hierarchy_path", [])
                )
            )
            out.append(
                RetrievalDoc(
                    doc_id=item["entry_id"],
                    library_name="standard_rule_library",
                    dimension="industry_type",
                    source_type="industry_item",
                    standard_no=standard_no,
                    effective_from=eff,
                    clause_id=table.get("clause_id"),
                    title=title,
                    text=text,
                    keywords=[k for k in keywords if k],
                    metadata=item,
                )
            )
        return out

    def _load_coding_rules(self, data: Dict[str, Any]) -> List[RetrievalDoc]:
        table = data.get("最终编码规则表", {})
        eff = self._effective_from(data, "GB 6441-2025", "2026-07-01")
        standard_no = "GB 6441-2025"
        out: List[RetrievalDoc] = []

        def add_doc(doc_id: str, title: str, text: str, metadata: Dict[str, Any], keywords: Iterable[str] = ()) -> None:
            out.append(
                RetrievalDoc(
                    doc_id=doc_id,
                    library_name="standard_rule_library",
                    dimension="coding",
                    source_type="coding_rule",
                    standard_no=standard_no,
                    effective_from=eff,
                    clause_id=",".join(table.get("clause_ids", [])) if table.get("clause_ids") else None,
                    title=title,
                    text=text,
                    keywords=[k for k in keywords if k],
                    metadata=metadata,
                )
            )

        code_structure = table.get("code_structure", {})
        add_doc(
            "coding::code_structure",
            "事故类型代码结构",
            join_non_empty(
                [
                    f"编码模式: {code_structure.get('pattern', '')}",
                    f"总长度: {code_structure.get('total_length', '')}",
                    f"分段: {json.dumps(code_structure.get('segments', []), ensure_ascii=False)}",
                    f"分类方式映射: {json.dumps(table.get('dimension_code_map', {}), ensure_ascii=False)}",
                ]
            ),
            code_structure,
            keywords=["WSA", "编码结构", "分类方式代码"],
        )
        add_doc(
            "coding::basic_type",
            "按基本事故类型编码规则",
            join_non_empty(
                [
                    json.dumps(table.get("basic_type_coding_rule", {}), ensure_ascii=False),
                    f"运行时顺序: {' -> '.join(table.get('runtime_generation_order', []))}",
                ]
            ),
            table.get("basic_type_coding_rule", {}),
            keywords=["WSA1", "基本事故类型", "0001", "0099"],
        )
        add_doc(
            "coding::injury_severity",
            "按人身伤害程度编码规则",
            json.dumps(table.get("injury_severity_coding_rule", {}), ensure_ascii=False),
            table.get("injury_severity_coding_rule", {}),
            keywords=["WSA2", "零伤亡", "轻伤", "重伤", "死亡"],
        )
        add_doc(
            "coding::industry",
            "按行业分类编码规则",
            join_non_empty(
                [
                    json.dumps(table.get("industry_coding_rule", {}), ensure_ascii=False),
                    f"补位示例: {json.dumps(table.get('industry_padding_examples', []), ensure_ascii=False)}",
                ]
            ),
            table.get("industry_coding_rule", {}),
            keywords=["WSA3", "GB/T4754", "去除门类后的4位编码"],
        )
        for idx, example in enumerate(table.get("examples", []), start=1):
            add_doc(
                f"coding::example::{idx}",
                f"编码示例 {idx}",
                json.dumps(example, ensure_ascii=False),
                example,
                keywords=[str(v) for v in example.values()],
            )
        return out
