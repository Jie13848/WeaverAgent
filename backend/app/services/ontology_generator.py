"""
本体生成服务
接口1：分析文本内容，生成适合社会模拟的实体和关系类型定义
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """将任意格式的名称转换为 PascalCase（如 'works_for' -> 'WorksFor', 'person' -> 'Person'）"""
    # 按非字母数字字符分割
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    # 再按 camelCase 边界分割（如 'camelCase' -> ['camel', 'Case']）
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    # 每个词首字母大写，过滤空串
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


# 本体生成的系统提示词
ONTOLOGY_SYSTEM_PROMPT = """你是一个专业的学术知识图谱本体设计专家。你的任务是分析给定的学术论文文本，设计适合**学术论文技术知识图谱**的实体类型和关系类型。

**重要：你必须输出有效的JSON格式数据，不要输出任何其他内容。**

## 核心任务背景

我们正在构建一个**学术论文技术知识图谱**。目标是从论文中提取：
- **技术/方法**（本文提出或使用的技术手段、算法、模型、框架）
- **创新点**（本文相对于已有工作的改进和突破）
- **效果/指标**（在哪些任务上取得了什么样的性能表现）
- **数据集**（实验使用的数据集）
- **任务**（论文解决的具体问题/任务）
- **对比方法**（baseline 方法）
- **作者与机构**（论文作者及其所属机构）

知识图谱用于后续 RAG 检索：用户提问"这篇论文用了什么方法""创新点是什么""在哪些数据集上测试的""效果如何"时，能精准召回相关节点和关系。

## 输出格式

请输出JSON格式，包含以下结构：

```json
{
    "entity_types": [
        {
            "name": "实体类型名称（英文，PascalCase）",
            "description": "简短描述（英文，不超过100字符）",
            "attributes": [
                {
                    "name": "属性名（英文，snake_case）",
                    "type": "text",
                    "description": "属性描述"
                }
            ],
            "examples": ["示例实体1", "示例实体2"]
        }
    ],
    "edge_types": [
        {
            "name": "关系类型名称（英文，UPPER_SNAKE_CASE）",
            "description": "简短描述（英文，不超过100字符）",
            "source_targets": [
                {"source": "源实体类型", "target": "目标实体类型"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "对论文内容的简要分析说明（中文，包括论文领域、主要技术和核心贡献）"
}
```

## 设计指南（极其重要！）

### 1. 实体类型设计 — 固定8个核心类型

**必须严格使用以下8个实体类型，不得增减，不得改名：**

1. **Paper** — 论文本身
   - 属性：`title`（论文标题）、`venue`（发表期刊/会议）、`pub_year`（发表年份）

2. **Method** — 本文提出或使用的技术方法/模型/算法/框架/模块
   - 包括：神经网络结构、注意力机制、损失函数、训练策略、推理算法等
   - 属性：`tech_type`（技术类别，如 "neural network / attention / training strategy"）、`detail`（一句话描述）

3. **Innovation** — 本文相对于已有工作的创新点/贡献/改进
   - 区别于 Method：Method 是"做了什么"，Innovation 是"比之前好在哪里"
   - 属性：`contribution_type`（类型，如 "architecture / efficiency / generalization"）、`detail`（一句话描述）

4. **Task** — 论文针对的具体 NLP/CV/ML 任务或应用场景
   - 例如：文本分类、机器翻译、目标检测、问答系统
   - 属性：`task_type`（任务大类）、`domain`（应用领域）

5. **Dataset** — 实验用到的数据集（包括训练集、测试集、基准集）
   - 属性：`scale`（数据规模）、`data_type`（数据类型，如 "text / image / multimodal"）

6. **Metric** — 评估指标（Accuracy、F1、BLEU、ROUGE、mAP 等）以及具体数值结果
   - 属性：`metric_value`（具体数值）、`comparison`（与基线相比的提升）

7. **Baseline** — 对比方法/baseline 模型（本文要超越的已有方法）
   - 属性：`venue`（该方法来自哪篇论文或哪个系统）、`detail`（一句话描述）

8. **Author** — 论文作者及其所属机构
   - 属性：`affiliation`（所属机构）、`role`（角色，如 "first author / corresponding author"）

**注意**：
- 属性名不能使用 `name`、`uuid`、`group_id`、`created_at`、`summary`（系统保留字）
- 使用 `title`、`detail`、`tech_type`、`venue` 等替代

### 2. 关系类型设计 — 必须正好8个，从以下固定集合中选择

**从下面列表中选择恰好8个，根据论文内容判断哪些关系最重要：**

候选关系（共12个，选8个）：
- `PROPOSES` — Paper → Method/Innovation（本文提出了...）
- `USES` — Paper/Method → Method/Dataset（使用/依赖了...）
- `EVALUATES_ON` — Paper → Dataset（在...数据集上评估）
- `ACHIEVES` — Paper/Method → Metric（取得了...指标结果）
- `OUTPERFORMS` — Method/Paper → Baseline（超越了...基线）
- `SOLVES` — Paper/Method → Task（解决了...任务）
- `IMPROVES_OVER` — Innovation → Baseline/Method（对...的改进）
- `COMPARED_WITH` — Paper → Baseline（与...方法对比）
- `AUTHORED_BY` — Paper → Author（由...撰写）
- `BUILDS_ON` — Method/Innovation → Method/Baseline（基于...构建）
- `APPLIED_TO` — Method → Task/Dataset（应用于...）
- `MEASURED_BY` — Task/Method → Metric（用...衡量效果）

**选择原则**：
- 优先选择在论文中频繁出现的关系
- 确保 `PROPOSES`、`EVALUATES_ON`、`ACHIEVES` 三个核心关系必须包含
- 其余5个根据论文内容灵活选择

### 3. source_targets 填写规则

每个关系的 source_targets 必须精确填写，只列出合理的源-目标组合，例如：
- `PROPOSES`: [{"source": "Paper", "target": "Method"}, {"source": "Paper", "target": "Innovation"}]
- `ACHIEVES`: [{"source": "Paper", "target": "Metric"}, {"source": "Method", "target": "Metric"}]
- `AUTHORED_BY`: [{"source": "Paper", "target": "Author"}]
"""


class OntologyGenerator:
    """
    本体生成器
    分析文本内容，生成实体和关系类型定义
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        analysis_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成本体定义
        
        Args:
            document_texts: 文档文本列表
            analysis_requirement: 分析需求描述
            additional_context: 额外上下文
            
        Returns:
            本体定义（entity_types, edge_types等）
        """
        # 构建用户消息
        user_message = self._build_user_message(
            document_texts, 
            analysis_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # 验证和后处理
        result = self._validate_and_process(result)
        
        return result
    
    # 传给 LLM 的文本最大长度（5万字）
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        analysis_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """构建用户消息"""
        
        # 合并文本
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # 如果文本超过5万字，截断（仅影响传给LLM的内容，不影响图谱构建）
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(原文共{original_length}字，已截取前{self.MAX_TEXT_LENGTH_FOR_LLM}字用于本体分析)..."
        
        message = f"""## 分析需求

{analysis_requirement}

## 论文内容

{combined_text}
"""
        
        if additional_context:
            message += f"""
## 额外说明

{additional_context}
"""
        
        message += """
请根据以上学术论文内容，设计知识图谱本体。

**必须遵守的规则**：
1. entity_types 必须正好8个，使用 system prompt 中规定的固定类型：Paper、Method、Innovation、Task、Dataset、Metric、Baseline、Author，顺序不变，名称不变
2. edge_types 必须正好8个，从候选12个关系中选择，其中 PROPOSES、EVALUATES_ON、ACHIEVES 三个必须包含
3. 每个关系的 source_targets 要填写准确，反映论文中实际存在的关系方向
4. analysis_summary 用中文总结：论文领域、核心方法、主要创新点、实验数据集和关键效果指标
5. 属性名不能使用 name、uuid、group_id、created_at、summary 等保留字
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证和后处理结果"""
        
        # 确保必要字段存在
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # 验证实体类型
        # 记录原始名称到 PascalCase 的映射，用于后续修正 edge 的 source_targets 引用
        entity_name_map = {}
        for entity in result["entity_types"]:
            # 强制将 entity name 转为 PascalCase（Zep API 要求）
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # 确保description不超过100字符
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # 验证关系类型
        for edge in result["edge_types"]:
            # 强制将 edge name 转为 SCREAMING_SNAKE_CASE（Zep API 要求）
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            # 修正 source_targets 中的实体名称引用，与转换后的 PascalCase 保持一致
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API 限制：最多 10 个自定义实体类型，最多 10 个自定义边类型
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # 去重：按 name 去重，保留首次出现的
        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        # 学术知识图谱的8个固定实体类型及默认定义
        REQUIRED_ACADEMIC_TYPES = {
            "Paper": {
                "name": "Paper",
                "description": "An academic paper or publication being analyzed.",
                "attributes": [
                    {"name": "title", "type": "text", "description": "Full title of the paper"},
                    {"name": "venue", "type": "text", "description": "Journal or conference where published"},
                    {"name": "pub_year", "type": "text", "description": "Year of publication"}
                ],
                "examples": ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers"]
            },
            "Method": {
                "name": "Method",
                "description": "A technical method, model, algorithm, or framework proposed or used in the paper.",
                "attributes": [
                    {"name": "tech_type", "type": "text", "description": "Category: neural network / attention / training strategy / etc."},
                    {"name": "detail", "type": "text", "description": "One-sentence description of the method"}
                ],
                "examples": ["Transformer", "BERT", "Cross-Attention", "Contrastive Loss"]
            },
            "Innovation": {
                "name": "Innovation",
                "description": "A specific contribution or improvement over prior work introduced by the paper.",
                "attributes": [
                    {"name": "contribution_type", "type": "text", "description": "Type: architecture / efficiency / generalization / etc."},
                    {"name": "detail", "type": "text", "description": "One-sentence description of the innovation"}
                ],
                "examples": ["eliminates recurrence for parallelization", "achieves state-of-the-art with less compute"]
            },
            "Task": {
                "name": "Task",
                "description": "A specific NLP/CV/ML task or application scenario addressed by the paper.",
                "attributes": [
                    {"name": "task_type", "type": "text", "description": "High-level category of the task"},
                    {"name": "domain", "type": "text", "description": "Application domain"}
                ],
                "examples": ["Machine Translation", "Text Classification", "Object Detection"]
            },
            "Dataset": {
                "name": "Dataset",
                "description": "A dataset used for training, evaluation, or benchmarking in the paper.",
                "attributes": [
                    {"name": "scale", "type": "text", "description": "Size or scale of the dataset"},
                    {"name": "data_type", "type": "text", "description": "Type: text / image / multimodal / etc."}
                ],
                "examples": ["WMT 2014 English-German", "ImageNet", "SQuAD"]
            },
            "Metric": {
                "name": "Metric",
                "description": "An evaluation metric and its measured value reported in the paper.",
                "attributes": [
                    {"name": "metric_value", "type": "text", "description": "Concrete numeric result"},
                    {"name": "comparison", "type": "text", "description": "Improvement over baseline"}
                ],
                "examples": ["BLEU 28.4", "F1 93.2", "Accuracy 95.1%"]
            },
            "Baseline": {
                "name": "Baseline",
                "description": "A prior method or model used as a comparison baseline in the paper.",
                "attributes": [
                    {"name": "venue", "type": "text", "description": "Paper or system the baseline comes from"},
                    {"name": "detail", "type": "text", "description": "One-sentence description"}
                ],
                "examples": ["RNN Seq2Seq", "ResNet-50", "GPT-2"]
            },
            "Author": {
                "name": "Author",
                "description": "An author of the paper and their institutional affiliation.",
                "attributes": [
                    {"name": "affiliation", "type": "text", "description": "Institution the author belongs to"},
                    {"name": "role", "type": "text", "description": "Role: first author / corresponding author / etc."}
                ],
                "examples": ["Ashish Vaswani", "Jacob Devlin"]
            }
        }

        # 确保所有8个固定类型都存在；LLM 返回的同名类型优先保留，缺失的用默认值补齐
        entity_names_present = {e["name"] for e in result["entity_types"]}
        for type_name, default_def in REQUIRED_ACADEMIC_TYPES.items():
            if type_name not in entity_names_present:
                logger.warning(f"Required academic entity type '{type_name}' missing, adding default.")
                result["entity_types"].append(default_def)

        # 最终截断到 MAX_ENTITY_TYPES（防御性）
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        # 必须包含的3个核心关系
        REQUIRED_EDGE_NAMES = {"PROPOSES", "EVALUATES_ON", "ACHIEVES"}
        present_edges = {e.get("name", "") for e in result["edge_types"]}
        missing_required = REQUIRED_EDGE_NAMES - present_edges

        REQUIRED_EDGE_DEFAULTS = {
            "PROPOSES": {
                "name": "PROPOSES",
                "description": "Paper proposes a method or innovation.",
                "source_targets": [
                    {"source": "Paper", "target": "Method"},
                    {"source": "Paper", "target": "Innovation"}
                ],
                "attributes": []
            },
            "EVALUATES_ON": {
                "name": "EVALUATES_ON",
                "description": "Paper or method is evaluated on a dataset.",
                "source_targets": [
                    {"source": "Paper", "target": "Dataset"},
                    {"source": "Method", "target": "Dataset"}
                ],
                "attributes": []
            },
            "ACHIEVES": {
                "name": "ACHIEVES",
                "description": "Paper or method achieves a performance metric result.",
                "source_targets": [
                    {"source": "Paper", "target": "Metric"},
                    {"source": "Method", "target": "Metric"}
                ],
                "attributes": []
            }
        }
        for edge_name in missing_required:
            logger.warning(f"Required edge type '{edge_name}' missing, adding default.")
            result["edge_types"].append(REQUIRED_EDGE_DEFAULTS[edge_name])

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将本体定义转换为Python代码（类似ontology.py）
        
        Args:
            ontology: 本体定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            '自定义实体类型定义',
            '由WeaverAgent自动生成，用于学术知识图谱构建',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== 实体类型定义 ==============',
            '',
        ]
        
        # 生成实体类型
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== 关系类型定义 ==============')
        code_lines.append('')
        
        # 生成关系类型
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # 转换为PascalCase类名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # 生成类型字典
        code_lines.append('# ============== 类型配置 ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # 生成边的source_targets映射
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

