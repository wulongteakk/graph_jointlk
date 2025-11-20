# -*- coding: utf-8 -*-
# @Time    : 2024/7/14 22:06
# @Author  : nongbin
# @FileName: llm.py
# @Software: PyCharm
# @Affiliation: tfswufe.edu.cn

import asyncio
import logging
import json

from src.shared.constants import (
    EXTRACTION_PROMPT_TEMPLATE, # 导入新的模板

)
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field, create_model

examples = [
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "WORKS_FOR",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "HAS_AWARD",
        "tail": "Best Talent",
        "tail_type": "Award",
    },
    {
        "text": (
            "Microsoft is a tech company that provide "
            "several products such as Microsoft Word"
        ),
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "PRODUCED_BY",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "lightweight app",
        "tail_type": "Characteristic",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "accessible offline",
        "tail_type": "Characteristic",
    },
]

system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrifing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'"
    "  - **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)


def get_dynamic_prompt_template(allowed_nodes: List[str], allowed_relationships: List[str]) -> str:
    """
    根据是否提供了节点和关系类型，生成不同的系统提示。
    """
    # 当用户在前端定义了节点和关系时的提示 (保持原始的严格指令)
    prompt_with_schema = (
        "You are a top-tier algorithm designed for extracting information in structured "
        "formats to build a knowledge graph. Your task is to identify entities and "
        "relationships from a given text. You must strictly follow the provided schema "
        "for node labels and relationship types. Do not infer any node labels or "
        "relationship types that are not explicitly provided in the schema.\n"
        "---------------------\n"
        "Schema:\n"
        f"Node Labels: {allowed_nodes}\n"
        f"Relationship Types: {allowed_relationships}\n"
        "---------------------\n"
    )
    # 当用户没有定义任何节点和关系时的提示 (给予模型更大的自由度)
    prompt_without_schema = (
        "You are a top-tier algorithm designed for extracting information in structured "
        "formats to build a knowledge graph. Your task is to identify the main entities and "
        "their relationships from a given text. First, analyze the text to determine the "
        "most relevant and logical schema (node labels and relationship types) for the "
        "information presented. Then, extract the entities and relationships based on "
        "the schema you have inferred. Focus on the core subjects, their defining "
        "attributes, and how they connect to one another."
    )

    # 如果用户提供了任何节点或关系类型，则使用严格的模板；否则，使用灵活的模板。
    if allowed_nodes or allowed_relationships:
        return prompt_with_schema
    else:
        return prompt_without_schema
def _get_additional_info(input_type: str) -> str:
    # Check if the input_type is one of the allowed values
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")

    # Perform actions based on the input_type
    if input_type == "node":
        return (
            "Ensure you use basic or elementary types for node labels.\n"
            "For example, when you identify an entity representing a person, "
            "always label it as **'Person'**. Avoid using more specific terms "
            "like 'Mathematician' or 'Scientist'"
        )
    elif input_type == "relationship":
        return (
            "Instead of using specific and momentary types such as "
            "'BECAME_PROFESSOR', use more general and timeless relationship types like "
            "'PROFESSOR'. However, do not sacrifice any accuracy for generality"
        )
    elif input_type == "property":
        return ""
    return ""


def optional_enum_field(
    enum_values: Optional[List[str]] = None,
    description: str = "",
    input_type: str = "node",
    **field_kwargs: Any,
) -> Any:
    """Utility function to conditionally create a field with an enum constraint."""
    if enum_values:
        return Field(
            ...,
            enum=enum_values,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    else:
        additional_info = _get_additional_info(input_type)
        return Field(..., description=description + additional_info, **field_kwargs)


class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )


def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, rel_types: Optional[List[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one of the types "
        "from the provided list in the user prompt.",
        f'The "head_type" key must contain the type of the extracted head entity, '
        f"which must be one of the types from {node_labels_str}."
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
        "Attempt to extract as many entities and relations as you can. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_prompt = PromptTemplate(
        template="""Based on the following example, extract entities and 
relations from the provided text.\n\n
Use the following entity types, don't use other entity that is not defined below:
# ENTITY TYPES:
{node_labels}

Use the following relation types, don't use other relation that is not defined below:
# RELATION TYPES:
{rel_types}

Below are a number of examples of text and their extracted entities and relationships.
{examples}

For the following text, extract entities and relations as in the provided example.
{format_instructions}\nText: {input}""",
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt


def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    node_properties: Union[bool, List[str]] = False,
) -> Type[_Graph]:
    """
    Simple model allows to limit node and/or relationship types.
    Doesn't have any node or relationship properties.
    """

    node_fields: Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            Field(..., description="Name or human-readable unique identifier."),
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the node.",
                input_type="node",
            ),
        ),
    }
    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                node_properties_mapped,
                description="Property key.",
                input_type="property",
            )
            value: str = Field(..., description="value")

        node_fields["properties"] = (
            Optional[List[Property]],
            Field(None, description="List of node properties"),
        )
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    class SimpleRelationship(BaseModel):
        """Represents a directed relationship between two nodes in a graph."""

        source_node_id: str = Field(
            description="Name or human-readable unique identifier of source node"
        )
        source_node_type: str = optional_enum_field(
            node_labels,
            description="The type or label of the source node.",
            input_type="node",
        )
        target_node_id: str = Field(
            description="Name or human-readable unique identifier of target node"
        )
        target_node_type: str = optional_enum_field(
            node_labels,
            description="The type or label of the target node.",
            input_type="node",
        )
        type: str = optional_enum_field(
            rel_types,
            description="The type of the relationship.",
            input_type="relationship",
        )

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""

        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(
            description="List of relationships"
        )

    return DynamicGraph


def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)


def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    return Relationship(source=source, target=target, type=rel.type)


def _parse_and_clean_json(
    argument_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    if 'Items' in argument_json["nodes"]:
        argument_json["nodes"] = argument_json['nodes']['Items']
    if 'Items' in argument_json["relationships"]:
        argument_json["relationships"] = argument_json["relationships"]['Items']
    nodes = []
    for node in argument_json["nodes"]:
        if not node.get("id"):  # Id is mandatory, skip this node
            continue
        nodes.append(
            Node(
                id=node["id"],
                type=node.get("type"),
            )
        )
    relationships = []
    for rel in argument_json["relationships"]:
        # Mandatory props
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

        # Node type copying if needed from node list
        if not rel.get("source_node_type"):
            try:
                rel["source_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["source_node_id"]
                ][0]
            except IndexError:
                rel["source_node_type"] = None
        if not rel.get("target_node_type"):
            try:
                rel["target_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["target_node_id"]
                ][0]
            except IndexError:
                rel["target_node_type"] = None

        source_node = Node(
            id=rel["source_node_id"],
            type=rel["source_node_type"],
        )
        target_node = Node(
            id=rel["target_node_id"],
            type=rel["target_node_type"],
        )
        relationships.append(
            Relationship(
                source=source_node,
                target=target_node,
                type=rel["type"],
            )
        )
    return nodes, relationships


def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize(),
            properties=el.properties,
        )
        for el in nodes
    ]


def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
        )
        for el in rels
    ]


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def _convert_to_graph_document(
    raw_schema: Dict[Any, Any],
) -> Tuple[List[Node], List[Relationship]]:
    # If there are validation errors
    if not raw_schema["parsed"]:
        try:
            try:  # OpenAI type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            except Exception:  # Google type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["function_call"]["arguments"]
                )

            nodes, relationships = _parse_and_clean_json(argument_json)
        except Exception:  # If we can't parse JSON
            return ([], [])
    else:  # If there are no validation errors use parsed pydantic object
        parsed_schema: _Graph = raw_schema["parsed"]
        nodes = (
            [map_to_base_node(node) for node in parsed_schema.nodes]
            if parsed_schema.nodes
            else []
        )

        relationships = (
            [map_to_base_relationship(rel) for rel in parsed_schema.relationships]
            if parsed_schema.relationships
            else []
        )
    # Title / Capitalize
    return _format_nodes(nodes), _format_relationships(relationships)


class LLMGraphTransformer:
    """Transform documents into graph-based documents using a LLM.

    It allows specifying constraints on the types of nodes and relationships to include
    in the output graph. The class doesn't support neither extract and node or
    relationship properties

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured
          output.
        allowed_nodes (List[str], optional): Specifies which node types are
          allowed in the graph. Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types
          are allowed in the graph. Defaults to an empty list, allowing all relationship
          types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to
          the LLM with additional instructions.
        strict_mode (bool, optional): Determines whether the transformer should apply
          filtering to strictly adhere to `allowed_nodes` and `allowed_relationships`.
          Defaults to True.

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm=ChatOpenAI(temperature=0)
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=["Person", "Organization"])

            doc = Document(page_content="Elon Musk is suing OpenAI")
            graph_documents = transformer.convert_to_graph_documents([doc])
    """
    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [],
        allowed_relationships: List[str] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        use_function_call: bool = True
    ) -> None:
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self._function_call = use_function_call
        # Check if the LLM really supports structured output
        try:
            llm.with_structured_output(_Graph)
        except NotImplementedError:
            self._function_call = False
        if not self._function_call:
            if node_properties:
                raise ValueError(
                    "The 'node_properties' parameter cannot be used "
                    "in combination with a LLM that doesn't support "
                    "native function calling."
                )
            try:
                import json_repair

                self.json_repair = json_repair
            except ImportError:
                raise ImportError(
                    "Could not import json_repair python package. "
                    "Please install it with `pip install json-repair`."
                )
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes, allowed_relationships
            )
            self.chain = prompt | llm
        else:
            # Define chain
            schema = create_simple_model(
                allowed_nodes, allowed_relationships, node_properties
            )
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = prompt or default_prompt
            self.chain = prompt | structured_llm

    def _fix_incomplete_json(self, json_str: str) -> str:
        """
        尝试修复不完整的JSON字符串，处理常见的LLM输出问题
        """
        # 去除代码块标记
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        # 步骤1: 尝试修复单引号问题
        if json_str.count("'") > json_str.count('"'):
            logging.info("检测到更多单引号，尝试转换为双引号")
            json_str = json_str.replace("'", '"')

        # 步骤2: 处理未完成的字符串
        # 寻找未闭合的字符串
        in_string = False
        escaped = False
        last_quote_pos = -1
        for i, char in enumerate(json_str):
            if escaped:
                escaped = False
                continue
            if char == '\\':
                escaped = True
            elif char == '"':
                in_string = not in_string
                last_quote_pos = i if in_string else last_quote_pos

        # 如果字符串未闭合，添加闭合引号
        if in_string:
            # 找到未闭合引号后，继续查找字母/数字/_的最后一位
            end_pos = last_quote_pos + 1
            while end_pos < len(json_str) and (json_str[end_pos].isalnum() or json_str[end_pos] == '_'):
                end_pos += 1
            # 更新 last_quote_pos 为字母/数字/_的最后一位
            last_quote_pos = end_pos - 1

            logging.info(f"发现未闭合的字符串，在位置{last_quote_pos}后添加引号")
            json_str = json_str[:last_quote_pos + 1] + '"' + json_str[last_quote_pos + 1:]

        # 步骤3: 处理未完成的数组或对象
        # 计算括号平衡
        open_brackets = 0
        open_braces = 0
        in_string = False
        escaped = False

        for char in json_str:
            if escaped:
                escaped = False
                continue
            if char == '\\':
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '[':
                    open_brackets += 1
                elif char == ']':
                    open_brackets -= 1
                elif char == '{':
                    open_braces += 1
                elif char == '}':
                    open_braces -= 1

        # 添加缺失的闭合括号
        if open_braces > 0:
            logging.info(f"添加{open_braces}个'}}'以闭合对象")
            json_str += '}' * open_braces
        if open_brackets > 0:
            logging.info(f"添加{open_brackets}个']'以闭合数组")
            json_str += ']' * open_brackets


        # 步骤4: 尝试修复不完整的最后一个对象
        # 寻找最后一个未闭合的对象
        last_open_brace = json_str.rfind('{')
        last_close_brace = json_str.rfind('}')

        if last_open_brace > last_close_brace:
            logging.info("发现未闭合的最后一个对象，尝试闭合")
            # 寻找最后一个键值对的结束
            last_colon = json_str.rfind(':', last_open_brace)
            if last_colon > -1:
                # 尝试找到值的结束
                last_value_char = max(
                    json_str.rfind(',', last_colon),
                    json_str.rfind(']', last_colon),
                    json_str.rfind('}', last_colon)
                )
                if last_value_char == -1:
                    # 没有找到值的结束，可能是不完整的值
                    logging.info("添加缺失的值和闭合括号")
                    json_str = json_str[:last_colon + 1] + ' "unknown"}'
                else:
                    # 闭合对象
                    logging.info("闭合不完整的对象")
                    json_str = json_str[:last_value_char + 1] + '}'

        return json_str

    def process_response(self, document: Document) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content

        logging.info(f"text是啥{text}")

        input_vars = {
            "input": text,
            "node_labels": str(self.allowed_nodes),
            "rel_types": str(self.allowed_relationships)
        }
        raw_schema = self.chain.invoke(input_vars)
        logging.info(f"LLM response: {raw_schema}")
        if self._function_call:

            raw_schema = cast(Dict[Any, Any], raw_schema)
            nodes, relationships = _convert_to_graph_document(raw_schema)
            logging.info(f"_function_call=true,nodes: {nodes}")
        else:
            nodes = []
            relationships = []

            try:
                # 尝试解析LLM的响应
                raw_content = raw_schema.content
                # 去除代码块标记
                if "```json" in raw_content:
                    raw_content = raw_content.split("```json")[1].split("```")[0]
                elif "```" in raw_content:
                    raw_content = raw_content.replace("```", "").strip()

                parsed_json = json.loads(raw_content)

            except json.JSONDecodeError as e:
                # 尝试修复不完整的JSON
                logging.error(f"原始JSON解析失败: {e}")
                logging.info("尝试修复不完整的JSON...")

                fixed_json_str = self._fix_incomplete_json(raw_schema.content)
                try:
                    parsed_json = json.loads(fixed_json_str)
                    logging.warning(f"成功修复JSON: {fixed_json_str[:100]}...")
                except Exception as e2:
                    logging.error(f"JSON修复失败: {e2}")
                    logging.error(f"原始LLM响应: {raw_schema.content}")
                    logging.error(f"修复后的JSON尝试: {fixed_json_str}")
                    return GraphDocument(nodes=[], relationships=[], source=document)



            nodes_set = set()
            for rel in parsed_json:
                if rel and "head" in rel and "head_type" in rel and "tail" in rel and "tail_type" in rel and "relation" in rel:
                    # 添加节点到集合中进行去重
                    nodes_set.add((rel["head"], rel["head_type"]))
                    nodes_set.add((rel["tail"], rel["tail_type"]))

                    # 创建关系
                    source_node = Node(id=rel["head"], type=rel["head_type"])
                    target_node = Node(id=rel["tail"], type=rel["tail_type"])
                    relationships.append(
                        Relationship(
                            source=source_node, target=target_node, type=rel["relation"]
                        )
                    )
                else:
                    logging.warning(f"Skipping invalid relation (old format): {rel}")
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
            logging.info(f"_function_call=false,nodes: {nodes}")


        # Strict mode filtering
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                       and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                       in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document]
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        return [self.process_response(document) for document in documents]


    async def aprocess_response(self, document: Document) -> GraphDocument:
        """
        Asynchronously processes a single document, transforming it into a
        graph document.
        """
        text = document.page_content
        input_vars = {
            "input": text,
            "node_labels": str(self.allowed_nodes),
            "rel_types": str(self.allowed_relationships)
        }
        raw_schema = await self.chain.ainvoke(input_vars)
        raw_schema = cast(Dict[Any, Any], raw_schema)
        nodes, relationships = _convert_to_graph_document(raw_schema)

        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]


            node_types = set(node.type for node in nodes)

            rel_types_set = set(rel.type for rel in relationships)
            rel_type_labels = {rel_type: i for i, rel_type in enumerate(rel_types_set)}

            for rel in relationships:
                label = rel_type_labels[rel.type]
                if rel.properties is None:
                    rel.properties = {}
                rel.properties['type_label'] = label

            document.metadata['node_types_count'] = len(node_types)
            document.metadata['relationship_types_count'] = len(rel_types_set)
            document.metadata['relationship_type_labels'] = rel_type_labels


        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document]
    ) -> List[GraphDocument]:
        """
        Asynchronously convert a sequence of documents into graph documents.
        """
        tasks = [
            asyncio.create_task(self.aprocess_response(document))
            for document in documents
        ]
        results = await asyncio.gather(*tasks)
        return results
