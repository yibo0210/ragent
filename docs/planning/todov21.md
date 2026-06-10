# Ragent AI v20.0

## Deep Research Engine

Enterprise Autonomous Research Platform

------

# 一、版本目标

将 Ragent AI 从：

```text
Question
  ↓
Answer
```

升级为：

```text
Research Goal
    ↓
Research Planning
    ↓
Evidence Collection
    ↓
Multi-Agent Investigation
    ↓
Evidence Verification
    ↓
Report Generation
```

使系统具备：

- 长时间任务执行（5~60分钟）
- 多阶段研究能力
- 自动补充检索
- 证据驱动推理
- 企业级研究报告生成

------

# 二、核心架构

新增独立执行链：

```text
Research Request
        │
        ▼
Research Planner
        │
        ▼
Research DAG
        │
        ▼
Research Executor
        │
 ┌──────┼────────┐
 │      │        │
 ▼      ▼        ▼

Web   Graph   Data
Agent Agent   Agent

 │      │       │
 └──────┼───────┘
        ▼

Evidence Store
        │
        ▼

Research Reviewer
        │
        ▼

Report Generator
        │
        ▼

Artifact Center
```

------

# 三、新增模块

## 1. Research Planner

路径：

```text
backend/research/planner.py
```

职责：

将用户目标转换为研究计划。

输入：

```text
分析中国AI Agent市场未来三年发展趋势
```

输出：

```yaml
research_goal:
  中国AI Agent市场未来三年发展趋势

tasks:

  - id: T1
    name: 市场规模分析

  - id: T2
    name: 技术路线分析

  - id: T3
    name: 竞争格局分析

  - id: T4
    name: 商业模式分析

  - id: T5
    name: 风险与机会分析
```

------

## 2. Research DAG

路径：

```text
backend/research/models.py
```

新增：

```python
ResearchPlan
ResearchTask
ResearchExecution
ResearchEvidence
ResearchReport
```

支持：

- 串行任务
- 并行任务
- 依赖关系

例如：

```text
T1
 │
 ├─────┐
 ▼     ▼

T2    T3

 └──┬──┘
    ▼

T4
 │
 ▼

T5
```

------

## 3. Research Executor

路径：

```text
backend/research/executor.py
```

职责：

执行整个研究任务。

基于现有：

```text
Workflow Executor
```

扩展：

```python
execute_task()

collect_evidence()

evaluate_completeness()

generate_report()
```

支持：

- 断点恢复
- 长时间运行
- Checkpoint

直接复用：

```text
LangGraph Checkpointer
```

------

## 4. Evidence Store

路径：

```text
backend/research/evidence_store.py
```

新增核心概念：

Evidence

而不是直接保存答案。

数据结构：

```python
class Evidence:

    id

    source

    content

    citation

    confidence

    task_id

    created_at
```

来源：

- GraphRAG
- Web Search
- SQL
- MCP
- Uploaded Documents

统一存储。

------

# 四、多 Agent 研究模式

现有：

```text
Supervisor
    ↓
Worker
```

升级：

```text
Research Coordinator
        ↓
Research Agents
```

------

## Web Research Agent

职责：

- 行业信息
- 新闻
- 报告

来源：

```text
Tavily
MCP Search
```

------

## Graph Research Agent

职责：

- 企业关系
- 技术关系
- 多跳推理

来源：

```text
GraphRAG
Reasoning Engine
```

------

## Data Research Agent

职责：

- SQL分析
- 财务数据
- KPI分析

来源：

```text
Data Analyst
```

------

## Internal Knowledge Agent

职责：

企业知识库调查

来源：

```text
Milvus
Neo4j
```

------

# 五、Research Reviewer

路径：

```text
backend/research/reviewer.py
```

职责：

判断研究是否充分。

检查：

```python
Evidence Coverage

Evidence Diversity

Citation Count

Confidence Score
```

------

评分：

```python
coverage_score

citation_score

confidence_score

completeness_score
```

------

例如：

```text
竞争格局分析

发现：

仅引用2个来源

结论：

证据不足
```

自动触发：

```text
Research Retry
```

再次检索。

------

# 六、自动补充研究

新增：

```text
Gap Analyzer
```

路径：

```text
backend/research/gap_analyzer.py
```

发现：

```text
缺失：
市场规模数据
```

自动生成：

```text
补充检索：
AI Agent Market Size 2025
```

进入：

```text
Research Loop
```

形成：

```text
Collect
→ Review
→ Gap Analysis
→ Collect
```

直到满足阈值。

------

# 七、研究报告生成器

路径：

```text
backend/research/report_generator.py
```

生成：

```markdown
# Executive Summary

# Key Findings

# Market Analysis

# Competitive Landscape

# Opportunities

# Risks

# Conclusion

# References
```

要求：

所有结论必须绑定：

```text
Evidence ID
Citation
Confidence
```

实现：

```text
Evidence Driven Report
```

而不是普通LLM写作。

------

# 八、Artifact升级

新增：

## PDF Artifact

```python
ResearchReport.pdf
```

------

## PPT Artifact

```python
ResearchSlides.pptx
```

------

## Executive Summary

```python
summary.md
```

------

## Evidence Package

```python
evidence.json
```

用于审计。

------

# 九、数据库设计

新增：

research_tasks

```sql
id
goal
status
created_at
```

------

research_executions

```sql
id
task_id
step_name
status
started_at
ended_at
```

------

research_evidence

```sql
id
task_id
source
citation
confidence
content
```

------

research_reports

```sql
id
task_id
artifact_path
summary
```

------

# 十、前端升级

新增：

Research Workspace

```text
Chat
│
├── Research
├── Evidence
├── Timeline
├── Reports
└── Artifacts
```

------

新增页面：

## Research Dashboard

显示：

```text
当前阶段

执行时间

Agent状态

证据数量

完成率
```

------

## Evidence Viewer

展示：

```text
来源

证据内容

引用

可信度
```

------

## Report Viewer

在线阅读：

```text
Markdown

PDF

PPT
```

------

# 十一、API

POST

```text
/research/create
```

创建任务

------

GET

```text
/research/{id}
```

查看状态

------

GET

```text
/research/{id}/evidence
```

查看证据

------

GET

```text
/research/{id}/report
```

查看报告

------

POST

```text
/research/{id}/cancel
```

取消任务

------

# 十二、实施计划

Phase 1（1周）

- Research Planner
- Research Models
- Research Task API

------

Phase 2（1周）

- Research Executor
- Evidence Store
- LangGraph Integration

------

Phase 3（1周）

- Reviewer
- Gap Analyzer
- Auto Retry

------

Phase 4（1周）

- Report Generator
- PDF/PPT Artifact

------

Phase 5（1周）

- Frontend Research Workspace
- Dashboard
- Evidence Viewer

------

# 预期效果

Ragent AI 将从：

```text
GraphRAG Assistant
```

升级为：

```text
Autonomous Research Platform
```

核心能力：

- 自动研究
- 自动补充检索
- 多Agent协作调查
- 证据驱动结论
- 企业级研究报告生成
- 长任务执行与审计追踪

```

```