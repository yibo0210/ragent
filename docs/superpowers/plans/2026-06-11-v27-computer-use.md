# v27 Computer Use Agent 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) for tracking.

**Goal:** 让 Agent 具备操作真实世界的能力。新增 Browser Agent（网页浏览/表单填写/数据抓取）、Terminal Agent（Shell 命令执行/文件操作/脚本运行）、Desktop Agent（文件系统/GUI 操作），通过 Observe→Plan→Act→Observe→Replan 循环实现自主计算机操作。最终实现"研究市场→生成报告→生成PPT→上传飞书→发送邮件"全自动完成。

**Architecture:** 新增 `backend/computer_use/` 包，包含 4 个模块。BrowserAgent 基于 Playwright 实现浏览器自动化（页面导航、元素定位、点击/输入、截图验证、数据提取），通过 `browser_action` tool 暴露给 Agent。TerminalAgent 在 Docker 沙箱中执行 Shell 命令，白名单 + 超时 + 输出截断三重安全防护。DesktopAgent 封装文件系统操作（读/写/列表/搜索）+ OS 级操作（打开应用/剪贴板/通知）。EnvironmentObserver 在每次动作后捕获环境变化（截图 diff、文件变更、进程状态），反馈给 Agent 决策下一步。所有 Computer Use 操作通过 MCP Server 暴露，复用 v9 MCP 基础设施。

**Tech Stack:** Playwright（浏览器自动化）· Docker SDK（沙箱执行）· MCP Python SDK（工具暴露）· Pillow（截图对比）· 复用 v9 MCPConnectionManager · 复用 v24 World State

---

## File Structure

```
backend/computer_use/                      # 新增包
├── __init__.py                            # 包入口 + 单例工厂
├── schemas.py                             # ComputerAction, ActionResult, EnvironmentSnapshot
├── browser_agent.py                       # Playwright 浏览器自动化
├── terminal_agent.py                      # Docker 沙箱 Shell 执行
├── desktop_agent.py                       # 文件系统 + OS 操作
├── environment_observer.py                # 环境变化检测 + 截图 diff

scripts/start_computer_use_mcp.py          # 新增: Computer Use MCP Server 启动脚本

tests/test_computer_use.py                 # 新增: Computer Use 测试
```

---

## Phase 1: Schemas + Browser Agent

### Task 1: Computer Use Schemas

**Files:**
- Create: `backend/computer_use/__init__.py`
- Create: `backend/computer_use/schemas.py`

```python
# backend/computer_use/schemas.py
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ActionType(str, Enum):
    # Browser actions
    BROWSER_NAVIGATE = "browser_navigate"
    BROWSER_CLICK = "browser_click"
    BROWSER_TYPE = "browser_type"
    BROWSER_SCROLL = "browser_scroll"
    BROWSER_SCREENSHOT = "browser_screenshot"
    BROWSER_EXTRACT = "browser_extract"

    # Terminal actions
    TERMINAL_EXEC = "terminal_exec"
    TERMINAL_READ_FILE = "terminal_read_file"
    TERMINAL_WRITE_FILE = "terminal_write_file"

    # Desktop actions
    DESKTOP_OPEN = "desktop_open"
    DESKTOP_SEARCH = "desktop_search"
    DESKTOP_NOTIFY = "desktop_notify"


class ActionResult(BaseModel):
    """Result of a computer use action."""
    action_id: str = ""
    action_type: ActionType
    success: bool = False
    output: str = ""                       # text output / extracted data
    screenshot_b64: str = ""               # base64 screenshot (browser actions)
    error: str = ""
    duration_ms: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EnvironmentSnapshot(BaseModel):
    """State of the environment after one or more actions."""
    snapshot_id: str = ""
    current_url: str = ""                  # browser
    page_title: str = ""
    visible_text: str = ""                 # extracted visible text
    screenshot_b64: str = ""
    recent_actions: list[ActionResult] = Field(default_factory=list)
    file_changes: list[str] = Field(default_factory=list)   # new/modified/deleted files
    running_processes: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ComputerUseTask(BaseModel):
    """A high-level computer use task."""
    task_id: str = ""
    goal: str = ""                         # "下载Tesla Q4财报PDF"
    steps: list[str] = Field(default_factory=list)
    max_steps: int = 20                    # safety limit
    allowed_domains: list[str] = Field(default_factory=list)  # browser whitelist
    allowed_commands: list[str] = Field(default_factory=list)  # terminal whitelist
    sandbox_enabled: bool = True
    timeout_seconds: int = 300
```

### Task 2: Browser Agent

**Files:**
- Create: `backend/computer_use/browser_agent.py`

```python
# backend/computer_use/browser_agent.py
"""BrowserAgent: Playwright-based web browser automation."""

from playwright.async_api import async_playwright

class BrowserAgent:
    async def navigate(self, url: str) -> ActionResult:
        """Navigate to URL and return page snapshot."""
        ...

    async def click(self, selector: str) -> ActionResult:
        """Click an element by CSS selector or text content."""
        ...

    async def type_text(self, selector: str, text: str) -> ActionResult:
        """Type text into an input field."""
        ...

    async def scroll(self, direction: str = "down", amount: int = 500) -> ActionResult:
        ...

    async def screenshot(self) -> ActionResult:
        """Capture full-page screenshot as base64."""
        ...

    async def extract_text(self, selector: str = "body") -> ActionResult:
        """Extract visible text content from the page."""
        ...

    async def extract_table(self, selector: str = "table") -> ActionResult:
        """Extract table data as JSON."""
        ...

    async def get_page_state(self) -> EnvironmentSnapshot:
        """Get current browser state for LLM observation."""
        ...

    # MCP tool wrappers — exposed as tools for agent
    def get_mcp_tools(self) -> list[dict]:
        """Return MCP tool schemas for browser actions."""
        return [
            {"name": "browser_navigate", "description": "Navigate to a URL", ...},
            {"name": "browser_click", "description": "Click an element", ...},
            {"name": "browser_type", "description": "Type text into input", ...},
            {"name": "browser_screenshot", "description": "Take page screenshot", ...},
            {"name": "browser_extract", "description": "Extract text/data from page", ...},
        ]
```

---

## Phase 2: Terminal + Desktop Agents

### Task 3: Terminal Agent + Desktop Agent

**Files:**
- Create: `backend/computer_use/terminal_agent.py`
- Create: `backend/computer_use/desktop_agent.py`

```python
# backend/computer_use/terminal_agent.py
"""TerminalAgent: sandboxed shell command execution via Docker."""

import subprocess
import docker

# Safety: command whitelist + dangerous pattern blacklist
_ALLOWED_COMMANDS = ["ls", "cat", "head", "tail", "wc", "grep", "find",
                      "python", "pip", "node", "curl", "wget", "git"]
_DANGEROUS_PATTERNS = ["rm -rf", "sudo", "chmod 777", "> /dev/", "mkfs",
                        "dd if=", ":(){ :|:& };:", "fork bomb"]

class TerminalAgent:
    def execute(
        self, command: str, cwd: str = "/workspace", timeout: int = 30
    ) -> ActionResult:
        """Execute a shell command in Docker sandbox.
        Safety: whitelist check → pattern scan → Docker exec → timeout → output truncation.
        """
        ...

    def read_file(self, path: str, max_lines: int = 200) -> ActionResult:
        """Safely read a file from the sandbox."""
        ...

    def write_file(self, path: str, content: str) -> ActionResult:
        """Write content to a file in the sandbox."""
        ...

    def list_dir(self, path: str = "/workspace") -> ActionResult:
        ...

    # MCP tool wrappers
    def get_mcp_tools(self) -> list[dict]:
        return [
            {"name": "terminal_exec", "description": "Execute a shell command", ...},
            {"name": "terminal_read_file", "description": "Read a file", ...},
            {"name": "terminal_write_file", "description": "Write a file", ...},
        ]


# backend/computer_use/desktop_agent.py
"""DesktopAgent: file system operations + OS-level actions."""

import os
import shutil
import platform

class DesktopAgent:
    def open_file(self, path: str) -> ActionResult:
        """Open a file with the default application."""
        ...

    def search_files(self, directory: str, pattern: str) -> ActionResult:
        """Search for files matching a pattern."""
        ...

    def get_file_info(self, path: str) -> ActionResult:
        """Get file metadata (size, modified time, type)."""
        ...

    def send_notification(self, title: str, message: str) -> ActionResult:
        """Send a desktop notification."""
        ...

    def get_clipboard(self) -> ActionResult:
        ...

    def set_clipboard(self, text: str) -> ActionResult:
        ...

    # MCP tool wrappers
    def get_mcp_tools(self) -> list[dict]:
        return [
            {"name": "desktop_open", "description": "Open a file or application", ...},
            {"name": "desktop_search", "description": "Search for files", ...},
            {"name": "desktop_notify", "description": "Send system notification", ...},
        ]
```

---

## Phase 3: Environment Observer + Control Loop

### Task 4: Environment Observer + Agent Loop

**Files:**
- Create: `backend/computer_use/environment_observer.py`
- Create: `scripts/start_computer_use_mcp.py`

```python
# backend/computer_use/environment_observer.py
"""EnvironmentObserver: captures environment changes for the Observe→Plan→Act loop."""

from PIL import Image
import base64, io

class EnvironmentObserver:
    def capture(self) -> EnvironmentSnapshot:
        """Capture current environment state (browser + terminal + desktop)."""
        ...

    def diff_screenshots(self, before: str, after: str) -> str:
        """Generate a text description of what changed between two screenshots.
        Uses pixel diff → bounding boxes → LLM description.
        """
        ...

    def detect_changes(self, before: EnvironmentSnapshot, after: EnvironmentSnapshot) -> list[str]:
        """Summarize what changed: URLs, visible text, files, processes."""
        changes = []
        if before.current_url != after.current_url:
            changes.append(f"页面跳转: {before.current_url} → {after.current_url}")
        if before.page_title != after.page_title:
            changes.append(f"标题变化: {before.page_title} → {after.page_title}")
        # ... more change detection
        return changes


# scripts/start_computer_use_mcp.py
"""MCP Server that exposes Computer Use tools to the Agent."""

import asyncio
from mcp.server import Server, stdio_server
from mcp.types import Tool, TextContent

from backend.computer_use.browser_agent import get_browser_agent
from backend.computer_use.terminal_agent import get_terminal_agent
from backend.computer_use.desktop_agent import get_desktop_agent

app = Server("ragent-computer-use")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        *[Tool(**t) for t in get_browser_agent().get_mcp_tools()],
        *[Tool(**t) for t in get_terminal_agent().get_mcp_tools()],
        *[Tool(**t) for t in get_desktop_agent().get_mcp_tools()],
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Dispatch to appropriate agent
    ...

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Task 5: Computer Use Agent Node

**Files:**
- Modify: `backend/agent/orchestrator.py` (add computer_use agent node)

```python
# v27: Computer Use Agent Node in orchestrator
# Observe → Plan → Act → Observe → Replan loop

async def computer_use_node(state: SupervisorState):
    observer = get_environment_observer()
    task = ComputerUseTask(**state.get("computer_use_task", {}))

    for step in range(task.max_steps):
        # 1. Observe: capture current environment state
        snapshot = observer.capture()

        # 2. Plan: LLM decides next action based on snapshot + goal
        next_action = await llm_plan_next_action(
            goal=task.goal,
            snapshot=snapshot,
            history=action_history,
        )

        # 3. Act: execute the chosen action
        result = await execute_action(next_action)

        # 4. Observe: capture changes
        new_snapshot = observer.capture()
        changes = observer.detect_changes(snapshot, new_snapshot)

        # 5. Check if goal achieved
        if await llm_check_goal_achieved(task.goal, new_snapshot, action_history):
            break
```

---

## Phase 4: Integration with Research

### Task 6: End-to-End Automation Pipeline

**Files:**
- Modify: `backend/research/executor.py`

完整自动化链路:

```python
# v27: End-to-end automation — research → report → PPT → upload
# In Research Executor, after report generation:

if state.get("auto_publish"):
    # Step 1: Generate PPT from report
    desktop_agent = get_desktop_agent()
    result = await terminal_agent.execute(
        "python scripts/generate_ppt.py --report research_report.md"
    )

    # Step 2: Upload to Feishu (via browser agent)
    browser = get_browser_agent()
    await browser.navigate("https://feishu.cn/drive/")
    await browser.click("上传文件")
    await browser.type_text("input[type=file]", f"/output/report.pptx")
    await browser.click("确认上传")

    # Step 3: Send email notification
    await browser.navigate("https://mail.feishu.cn/")
    await browser.click("写邮件")
    await browser.type_text("#subject", f"研究报告: {plan.goal}")
    await browser.type_text("#body", f"报告已完成，请查看: {feishu_link}")
    await browser.click("发送")

    # Step 4: Desktop notification
    desktop_agent.send_notification("研究完成", f"报告已上传至飞书，邮件已发送")
```

---

## Self-Review

| 610 文档 v27 需求 | 覆盖 |
|---|---|
| Browser Agent (Playwright 自动化) | Task 2 (browser_agent.py) |
| Terminal Agent (Docker 沙箱) | Task 3 (terminal_agent.py) |
| Desktop Agent (文件系统/通知) | Task 3 (desktop_agent.py) |
| EnvironmentObserver (截图 diff + 变更检测) | Task 4 (environment_observer.py) |
| Computer Use MCP Server | Task 4 (start_computer_use_mcp.py) |
| Observe→Plan→Act→Replan 循环 | Task 5 (computer_use_node) |
| 端到端自动化 (研究→PPT→上传→邮件) | Task 6 (executor.py) |
| 安全三重防护 (白名单+Docker+超时) | Task 3 (terminal_agent.py) |
