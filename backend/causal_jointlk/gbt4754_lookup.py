from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class GBT4754Entry:
    full_code: str
    type_code: str
    section_code: str
    section_name: str
    division_code: str
    division_name: str
    group_code: str
    group_name: str
    class_code: str
    class_name: str
    full_name: str
    aliases: List[str]
    keywords: List[str]

    @classmethod
    def from_dict(cls, row: Dict[str, object]) -> "GBT4754Entry":
        return cls(
            full_code=str(row.get("full_code", "")),
            type_code=str(row.get("type_code", "")),
            section_code=str(row.get("section_code", "")),
            section_name=str(row.get("section_name", "")),
            division_code=str(row.get("division_code", "")),
            division_name=str(row.get("division_name", "")),
            group_code=str(row.get("group_code", "")),
            group_name=str(row.get("group_name", "")),
            class_code=str(row.get("class_code", "")),
            class_name=str(row.get("class_name", "")),
            full_name=str(row.get("full_name", "")),
            aliases=list(row.get("aliases", []) or []),
            keywords=list(row.get("keywords", []) or []),
        )


class GBT4754Lookup:
    def __init__(
        self,
        code_file: str = "configs/standards/gbt4754_2017_codes.jsonl",
        alias_file: str = "configs/standards/gbt4754_aliases.json",
    ) -> None:
        self.code_file = Path(code_file)
        if not self.code_file.exists():
            sample = self.code_file.with_suffix(".sample.jsonl")
            if sample.exists():
                self.code_file = sample
        self.alias_file = Path(alias_file)
        self.entries: List[GBT4754Entry] = list(self._load_entries(self.code_file))
        self.alias_map: Dict[str, List[str]] = self._load_aliases(self.alias_file)
        self.by_full_code = {e.full_code.upper(): e for e in self.entries if e.full_code}
        self.by_type_code = {e.type_code: e for e in self.entries if e.type_code}
        self.by_name = {e.full_name: e for e in self.entries if e.full_name}

    def _load_entries(self, path: Path) -> Iterable[GBT4754Entry]:
        if not path.exists():
            return []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            yield GBT4754Entry.from_dict(json.loads(raw))

    def _load_aliases(self, path: Path) -> Dict[str, List[str]]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def lookup_by_code(self, code: str) -> Optional[GBT4754Entry]:
        norm = (code or "").strip().upper()
        if not norm:
            return None
        if norm in self.by_full_code:
            return self.by_full_code[norm]
        digits = "".join(ch for ch in norm if ch.isdigit())
        return self.by_type_code.get(digits)

    def lookup_by_name(self, name: str) -> Optional[GBT4754Entry]:
        norm = (name or "").strip()
        if not norm:
            return None
        if norm in self.by_name:
            return self.by_name[norm]
        for entry in self.entries:
            if norm == entry.class_name or norm == entry.division_name or norm == entry.group_name:
                return entry
        return None

    def expand_alias(self, token: str) -> List[str]:
        return list(self.alias_map.get(token, []))

    def lookup_candidates_from_text(self, text: str, top_k: int = 5) -> List[Dict[str, object]]:
        text = (text or "").strip()
        if not text:
            return []
        hits: List[Dict[str, object]] = []
        seen = set()

        for token in self._extract_code_candidates(text):
            entry = self.lookup_by_code(token)
            if entry and entry.full_code not in seen:
                seen.add(entry.full_code)
                hits.append(self._pack(entry, 3.0, f"exact_code:{token}"))

        for entry in self.entries:
            score = 0.0
            rules: List[str] = []
            if entry.full_name and entry.full_name in text:
                score += 2.2
                rules.append(f"exact_name:{entry.full_name}")
            if entry.class_name and entry.class_name and entry.class_name in text:
                score += 1.8
                rules.append(f"class_name:{entry.class_name}")
            for alias in entry.aliases:
                if alias and alias in text:
                    score += 1.2
                    rules.append(f"alias:{alias}")
            for kw in entry.keywords:
                if kw and kw in text:
                    score += 0.35
                    rules.append(f"keyword:{kw}")
            if score > 0 and entry.full_code not in seen:
                seen.add(entry.full_code)
                hits.append(self._pack(entry, score, *rules))

        for alias, targets in self.alias_map.items():
            if alias and alias in text:
                for target in targets:
                    entry = self.lookup_by_code(target) or self.lookup_by_name(target)
                    if entry and entry.full_code not in seen:
                        seen.add(entry.full_code)
                        hits.append(self._pack(entry, 1.1, f"alias_map:{alias}->{target}"))

        hits.sort(key=lambda x: float(x["score"]), reverse=True)
        return hits[:top_k]

    def hierarchical_backoff(self, entry: GBT4754Entry) -> Dict[str, str]:
        return {
            "section_code": entry.section_code,
            "division_code": entry.division_code,
            "group_code": entry.group_code,
            "class_code": entry.class_code,
        }

    def _extract_code_candidates(self, text: str) -> Sequence[str]:
        tokens = []
        current = []
        for ch in text:
            if ch.isalnum():
                current.append(ch)
            else:
                if current:
                    tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return [tok for tok in tokens if any(c.isdigit() for c in tok) and 2 <= len(tok) <= 5]

    def _pack(self, entry: GBT4754Entry, score: float, *rules: str) -> Dict[str, object]:
        return {
            "entry": entry,
            "score": score,
            "rule_hits": [r for r in rules if r],
        }
