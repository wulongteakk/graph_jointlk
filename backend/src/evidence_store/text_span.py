import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class TextSpan:
    """一个文本片段在原文中的字符跨度（start/end 为 Python 字符索引）。"""

    start: int
    end: int
    text: str


# -----------------------------------------------------------------------------
# Sentence splitting
# -----------------------------------------------------------------------------

# 中文标点 + 换行的启发式分句
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])\s*")


def split_sentences_with_offsets(text: str, max_len: int = 400) -> List[TextSpan]:
    """启发式分句（中文标点 + 换行），并返回每句在原文中的字符跨度。

    说明：
    - 该函数不追求 NLP 级别严格性，而是用于“证据定位”的轻量切分。
    - 对于过长句子，会进行二次软切/硬切，保证每个 EvidenceUnit 不超过 max_len。
    """

    if not text:
        return []

    spans: List[TextSpan] = []
    cursor = 0

    parts = _SENTENCE_SPLIT_RE.split(text)
    for part in parts:
        if not part or not part.strip():
            continue

        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        cursor = end

        if max_len and (end - start) > max_len:
            spans.extend(_split_long_segment(seg=text[start:end], base_start=start, max_len=max_len))
        else:
            spans.append(TextSpan(start=start, end=end, text=part))

    return [s for s in spans if s.text and s.text.strip()]


def _split_long_segment(seg: str, base_start: int, max_len: int) -> List[TextSpan]:
    soft_re = re.compile(r"(?<=[，,、：:])\s*")
    parts = soft_re.split(seg)

    spans: List[TextSpan] = []
    cursor = 0
    for p in parts:
        if not p or not p.strip():
            continue
        s = seg.find(p, cursor)
        if s < 0:
            s = cursor
        e = s + len(p)
        cursor = e

        if (e - s) > max_len:
            # 硬切
            i = s
            while i < e:
                j = min(e, i + max_len)
                spans.append(TextSpan(start=base_start + i, end=base_start + j, text=seg[i:j]))
                i = j
        else:
            spans.append(TextSpan(start=base_start + s, end=base_start + e, text=p))

    return [x for x in spans if x.text and x.text.strip()]


def pick_span_sentence(spans: List[TextSpan], start: Optional[int], end: Optional[int]) -> int:
    """给定字符跨度，选它落在哪个句子里。找不到就返回 0。"""
    if not spans:
        return 0
    if start is None:
        return 0
    for i, sp in enumerate(spans):
        if sp.start <= start < sp.end:
            return i
    return 0


# -----------------------------------------------------------------------------
# Paragraph splitting
# -----------------------------------------------------------------------------

# 常见标题：
#  1) 1. / 1.1 / 1.1.1
#  2) 一、 二、 三、
#  3) （一） （二）
#  4) 【标题】
#  5) 短句以“：”结尾
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:\d+(?:\.\d+){0,4})"  # 1 / 1.1 / 1.1.1
    r"|(?:[一二三四五六七八九十]{1,3})"  # 一/二/三
    r"|(?:第[一二三四五六七八九十\d]{1,3})"  # 第三/第3
    r"|(?:\([一二三四五六七八九十]{1,3}\))"  # (一)
    r"|(?:（[一二三四五六七八九十]{1,3}）)"  # （一）
    r")"
    r"[\.、:：\s]+\S+"
)

_BRACKET_HEADING_RE = re.compile(r"^\s*[【\[].{1,60}[】\]]\s*$")

# 列表项 / 项目符号
_BULLET_RE = re.compile(r"^\s*(?:[\-\*•·]|\d+\)|\d+、|[a-zA-Z]\)|[①②③④⑤⑥⑦⑧⑨⑩])\s+\S+")

# 典型中文段落缩进（全角空格）或多个半角空格/tab
_INDENT_RE = re.compile(r"^(?:\u3000\u3000|\s{2,}|\t)\S+")


def split_paragraphs_with_offsets(text: str, max_len: int = 1200) -> List[TextSpan]:
    """按“空行/标题/缩进/项目符号”启发式切段，并返回每段在原文中的字符跨度。

    设计目标：
    - 适配报告/纪要类文本：存在标题行、列表项、空行。
    - 对 PDF 抽取的“强换行”较多场景：尽量避免把每一行都当段落。

    规则（优先级从高到低）：
    1) 空行分段：\n\s*\n
    2) 标题行分段：数字编号/中文编号/括号编号/【标题】/短句冒号
    3) 列表项分段：- * • 1) 1、 ①
    4) 缩进分段：行首明显缩进（\u3000\u3000 或 >=2 空格 或 tab），且上一段较完整

    注意：
    - 段落过长会做二次切分，避免 EvidenceUnit 过大。
    """

    if not text:
        return []

    # 保留换行以便 offset 精确
    lines = text.splitlines(keepends=True)
    spans: List[TextSpan] = []

    para_start: Optional[int] = None
    para_end: Optional[int] = None
    offset = 0

    prev_line_stripped = ""
    prev_was_blank = True

    def flush(end_offset: int):
        nonlocal para_start, para_end
        if para_start is None:
            return
        if end_offset <= para_start:
            para_start = None
            para_end = None
            return
        seg = text[para_start:end_offset]
        if seg and seg.strip():
            # 对超长段落进行二次切分
            if max_len and len(seg) > max_len:
                spans.extend(_split_long_paragraph(seg, base_start=para_start, max_len=max_len))
            else:
                spans.append(TextSpan(start=para_start, end=end_offset, text=seg))
        para_start = None
        para_end = None

    for line in lines:
        line_start = offset
        offset += len(line)

        raw = line.rstrip("\r\n")
        stripped = raw.strip()

        # 1) 空行
        if stripped == "":
            flush(line_start)
            prev_line_stripped = ""
            prev_was_blank = True
            continue

        # 判断是否“标题/列表/缩进”起始
        is_heading = _is_heading_line(stripped)
        is_bullet = bool(_BULLET_RE.match(stripped))
        is_indent = bool(_INDENT_RE.match(raw))  # 用 raw 判断更准确（保留缩进）

        # 2) 标题或列表项：强制新段落
        if is_heading or is_bullet:
            flush(line_start)
            para_start = line_start
            prev_was_blank = False
            prev_line_stripped = stripped
            continue

        # 3) 缩进：条件触发新段落
        # - 如果上一行是空行：当然新段落
        # - 或者上一行看起来“结束了”（。！？；）且当前行缩进明显
        # - 或者上一行是标题样式
        if is_indent:
            if prev_was_blank or _looks_like_para_end(prev_line_stripped) or _is_heading_line(prev_line_stripped):
                flush(line_start)
                para_start = line_start
                prev_was_blank = False
                prev_line_stripped = stripped
                continue

        # 普通行：延续当前段落
        if para_start is None:
            para_start = line_start

        prev_was_blank = False
        prev_line_stripped = stripped

    # flush last
    flush(len(text))

    # 清理空白段
    return [s for s in spans if s.text and s.text.strip()]


def _split_long_paragraph(seg: str, base_start: int, max_len: int) -> List[TextSpan]:
    """对超长段落做二次切分。

    优先在换行、分号、句号处切；否则硬切。
    """

    # 先按换行拆
    parts = re.split(r"(?<=\n)", seg)
    spans: List[TextSpan] = []
    cursor = 0

    buf = ""
    buf_start = 0

    def flush_buf():
        nonlocal buf, buf_start
        if buf and buf.strip():
            spans.append(TextSpan(start=base_start + buf_start, end=base_start + buf_start + len(buf), text=buf))
        buf = ""
        buf_start = cursor

    for p in parts:
        if not p:
            continue
        if not buf:
            buf_start = cursor
        if len(buf) + len(p) <= max_len:
            buf += p
        else:
            flush_buf()
            buf_start = cursor
            buf = p
        cursor += len(p)

    flush_buf()

    # 若仍有超长块，再按标点软切/硬切
    out: List[TextSpan] = []
    for sp in spans:
        if len(sp.text) <= max_len:
            out.append(sp)
        else:
            out.extend(_split_long_segment(seg=sp.text, base_start=sp.start, max_len=max_len))

    return [x for x in out if x.text and x.text.strip()]


def pick_span_paragraph(spans: List[TextSpan], start: Optional[int], end: Optional[int]) -> int:
    """给定字符跨度，选它落在哪个段落里。找不到就返回 0。"""
    if not spans:
        return 0
    if start is None:
        return 0
    for i, sp in enumerate(spans):
        if sp.start <= start < sp.end:
            return i
    return 0


def _is_heading_line(stripped_line: str) -> bool:
    if not stripped_line:
        return False
    if len(stripped_line) <= 2:
        return False
    if _BRACKET_HEADING_RE.match(stripped_line):
        return True
    if _HEADING_RE.match(stripped_line) and len(stripped_line) <= 80:
        return True

    # 短句以冒号结尾：可能是小标题
    if len(stripped_line) <= 40 and stripped_line.endswith((":", "：")):
        return True

    # 常见报告标题词（可扩展）
    keywords = (
        "事故经过",
        "事故原因",
        "原因分析",
        "调查情况",
        "整改措施",
        "处理意见",
        "责任认定",
        "工程概况",
        "基本情况",
        "现场情况",
        "结论",
    )
    if any(stripped_line.startswith(k) for k in keywords) and len(stripped_line) <= 60:
        return True

    return False


def _looks_like_para_end(prev_line_stripped: str) -> bool:
    if not prev_line_stripped:
        return False
    return prev_line_stripped.endswith(("。", "！", "？", "；", ".", "!", "?", ";"))


# -----------------------------------------------------------------------------
# Span match
# -----------------------------------------------------------------------------


def find_best_span(text: str, snippet: str) -> Tuple[Optional[int], Optional[int]]:
    """在 text 中定位 snippet，返回 (start,end)。找不到返回 (None,None)。"""
    if not text or not snippet:
        return (None, None)

    s = text.find(snippet)
    if s >= 0:
        return (s, s + len(snippet))

    # 兜底：压缩空白
    norm_text = re.sub(r"\s+", " ", text)
    norm_snip = re.sub(r"\s+", " ", snippet)
    s2 = norm_text.find(norm_snip)
    if s2 < 0:
        return (None, None)

    # 近似定位：在原文找一个短 anchor
    anchor = norm_snip[:12].strip()
    if not anchor:
        return (None, None)
    s3 = text.find(anchor)
    if s3 < 0:
        return (None, None)
    e3 = min(len(text), s3 + len(norm_snip))
    return (s3, e3)
