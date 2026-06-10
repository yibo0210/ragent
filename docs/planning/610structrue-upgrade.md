# Ragent 2.0 架构升级路线图

## 愿景

将 Ragent 从：

```text
Enterprise GraphRAG Platform
```

升级为：

```text
Research-Oriented General Agent Platform
```

最终演进为：

```text
General Agent Operating System
```

核心目标：

- 不推翻现有架构
- 不重写 Workflow
- 不重写 GraphRAG
- 不重写 Memory

而是在现有基础上增加：

- Dynamic Research
- Episodic Memory
- Agent Learning
- World State
- Artifact Lifecycle
- Computer Use

使 Agent 具备：

- 自主研究能力
- 长周期执行能力
- 经验积累能力
- 多轮决策能力
- 持续运行能力

------

# 一、现状分析

## 当前优势

Ragent 已具备：

### Knowledge Layer

- GraphRAG
- Hybrid Retrieval
- Knowledge Graph
- Semantic Search

### Workflow Layer

- LangGraph Workflow
- Supervisor Architecture
- Planner
- Critic
- HITL

### Agent Layer

- Multi-Agent
- Tool Calling
- MCP Integration

### Infrastructure Layer

- Memory
- Billing
- Observability
- RBAC
- Artifact Storage

已经具备 Agent Platform 的基础设施。

------

## 当前瓶颈

### 1. Plan Driven

当前：

```text
Plan
→ Execute
→ Finish
```

Agent 无法持续探索。

------

### 2. 缺乏经验积累

Agent 不会因为过去任务变得更聪明。

------

### 3. 缺乏环境认知

Agent 只理解对话。

不理解真实工作环境。

------

### 4. 缺乏长期执行

Agent 生命周期仅存在于当前 Workflow。

------

### 5. 缺乏世界模型

Agent 无法维护：

- 文件状态
- 浏览器状态
- 任务状态
- Artifact 状态

------

# 二、升级路线总览

## Phase 1

Research Agent

目标：

```text
从任务执行器
升级为研究者
```

版本：

```text
V21
```

------

## Phase 2

Learning Agent

目标：

```text
从会研究
升级为会成长
```

版本：

```text
V22-V23
```

------

## Phase 3

Agent OS

目标：

```text
从会成长
升级为会长期工作
```

版本：

```text
V24-V25
```

------

## Phase 4

General Agent

目标：

```text
从知识Agent
升级为通用Agent
```

版本：

```text
V26-V27
```

------

# 三、V21 Dynamic Research Architecture

## 核心思想

从：

```text
Research
→ Result
```

升级为：

```text
Hypothesis
→ Evidence
→ Verification
→ New Question
→ Research
```

------

## 新增模块

```text
research/

├── hypothesis_generator.py
├── evidence_graph.py
├── conflict_detector.py
├── question_expander.py
├── confidence_estimator.py
```

------

## Research Loop

```text
Goal
 ↓
Hypothesis Generation
 ↓
Research
 ↓
Evidence Collection
 ↓
Conflict Detection
 ↓
Question Expansion
 ↓
Research
```

------

## Example

用户：

```text
为什么Intel AI失败？
```

Agent：

```text
Hypothesis A
组织结构问题

Hypothesis B
CUDA生态问题

Hypothesis C
资本投入不足
```

Agent 分别验证。

形成：

```text
Evidence Graph
```

而非：

```text
Evidence List
```

------

## Deliverables

- Dynamic Research Loop
- Evidence Graph
- Hypothesis Engine
- Multi-Round Investigation

------

# 四、V22 Episodic Memory

## 当前 Memory

```text
Fact
Preference
Task
Relation
```

属于：

```text
Semantic Memory
```

------

## 新增 Episodic Memory

记录：

```text
Goal
Plan
Actions
Failures
Outcome
Lessons
```

------

## Memory Schema

Episode

```json
{
  "goal": "",
  "plan": [],
  "actions": [],
  "failures": [],
  "outcome": "",
  "lessons": []
}
```

------

## Example

任务：

```text
分析Tesla财报
```

经验：

```text
SEC Filing优先级最高
```

未来任务：

```text
自动先检索SEC
```

------

## 新增模块

```text
memory/

├── episodic_store.py
├── lesson_extractor.py
├── episode_retriever.py
├── memory_ranker.py
```

------

# 五、V23 Agent Learning Loop

## 核心思想

从：

```text
Critique
→ Replan
```

升级为：

```text
Critique
→ Root Cause
→ Lesson
→ Memory
→ Future Improvement
```

------

## 新增模块

```text
learning/

├── failure_analyzer.py
├── lesson_generator.py
├── strategy_library.py
├── policy_store.py
```

------

## Example

SQL执行失败。

Agent发现：

```text
Schema理解错误
```

生成：

```text
Policy:
执行SQL前必须Schema Discovery
```

写入 Memory。

------

## 长期效果

Agent 将形成：

```text
Execution Policy Library
```

持续优化。

------

# 六、V24 World State Architecture

## 核心思想

Agent 必须理解世界。

不仅理解对话。

------

## 新增 World State

```python
class WorldState:

    current_goal

    active_tasks

    resources

    artifacts

    open_contexts

    environment
```

------

## 新增模块

```text
world/

├── state_manager.py
├── artifact_registry.py
├── resource_registry.py
├── context_manager.py
```

------

## Agent行为

Agent始终知道：

```text
当前做什么

做到哪一步

已经产出什么

下一步是什么
```

------

# 七、V25 Artifact-Centric Agent

## 核心思想

未来 Agent 的核心不是对话。

而是 Artifact。

------

## Artifact Lifecycle

```text
Create
↓
Version
↓
Update
↓
Review
↓
Publish
```

------

## Artifact Model

```json
{
  "id": "",
  "type": "",
  "version": "",
  "status": "",
  "dependencies": []
}
```

------

## Example

```text
市场报告
```

成为：

```text
Artifact #42
```

后续：

```text
更新42号报告
```

Agent 自动接续工作。

------

## 新增模块

```text
artifact/

├── artifact_manager.py
├── version_manager.py
├── dependency_graph.py
├── artifact_memory.py
```

------

# 八、V26 Decentralized Agent Network

## 当前

```text
Supervisor
 ↓
Workers
```

------

## 升级

```text
Research Agent
 ↔
Graph Agent
 ↔
Data Agent
 ↔
Web Agent
```

------

## Agent Bus

```text
agent_bus/

├── mailbox.py
├── message_router.py
├── event_bus.py
├── negotiation.py
```

------

## 能力

Agent可主动：

```text
请求帮助

共享证据

共享记忆

共享策略
```

------

# 九、V27 Computer Use Agent

## 目标

让 Agent 操作真实世界。

------

## 新增 Agent

```text
Browser Agent

Desktop Agent

Terminal Agent
```

------

## Computer Use Loop

```text
Observe
↓
Plan
↓
Act
↓
Observe
↓
Replan
```

------

## Example

```text
研究市场
↓
生成报告
↓
生成PPT
↓
上传飞书
↓
发送邮件
```

全自动完成。

------

## 新增模块

```text
computer_use/

├── browser_agent.py
├── desktop_agent.py
├── terminal_agent.py
├── environment_observer.py
```

------

# 十、最终架构

## Ragent 3.0

```text
                    Goal
                      │
                      ▼
               Reasoning Core
                      │
                      ▼
             Dynamic Planner
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼

    Research      Graph       Data
      Agent       Agent      Agent

         ▼            ▼            ▼
              Agent Message Bus

                      │
                      ▼

               Shared Memory

       ┌────────────┼────────────┐
       ▼            ▼            ▼

 Episodic     Semantic      Policy
  Memory       Memory       Memory

                      │
                      ▼

                World State

                      │
                      ▼

              Artifact System

                      │
                      ▼

              Computer Use

                      │
                      ▼

                 Deliverables
```

------

# 实施优先级

P0（立即开始）

- Dynamic Research
- Evidence Graph
- Hypothesis Engine

P1

- Episodic Memory
- Agent Learning Loop

P2

- World State
- Artifact Lifecycle

P3

- Computer Use
- Decentralized Agent Network

------

# 最终目标

将 Ragent 从：

```text
Enterprise GraphRAG Platform
```

升级为：

```text
Research-Oriented General Agent Platform
```

最终成为：

```text
Agent Operating System
```

具备：

- 深度研究能力
- 长周期任务能力
- 自我学习能力
- 环境感知能力
- Artifact驱动能力
- Computer Use能力
- 多Agent协作能力