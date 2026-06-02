"""MCP 连接管理器

维护与多个 MCP Server 的连接，提供工具发现和调用能力。
支持 SSE（HTTP）和 stdio（本地进程）两种传输方式。
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

log = logging.getLogger(__name__)


class MCPConnection:
    """单个 MCP Server 连接。"""

    def __init__(self, server_name: str, url: str, transport: str = "sse"):
        self.server_name = server_name
        self.url = url
        self.transport = transport
        self.session: ClientSession | None = None
        self.tools: list[dict] = []
        self._context = None
        self._streams = None

    async def connect(self, args: list[str] = None) -> bool:
        """建立连接并初始化会话。

        Args:
            args: stdio 模式下的命令参数，如 ["scripts/test_mcp_server.py"]
        """
        try:
            if self.transport == "sse":
                self._context = sse_client(url=self.url)
                self._streams = await self._context.__aenter__()
            elif self.transport == "stdio":
                cmd_args = args or []
                params = StdioServerParameters(command=self.url, args=cmd_args)
                self._context = stdio_client(params)
                self._streams = await self._context.__aenter__()
            else:
                raise ValueError(f"不支持的传输方式: {self.transport}")

            read_stream, write_stream = self._streams
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            # 拉取工具列表
            result = await self.session.list_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                }
                for tool in result.tools
            ]
            log.info("MCP 连接成功: %s, 工具数: %d", self.server_name, len(self.tools))
            return True
        except Exception as e:
            log.error("MCP 连接失败: %s, 错误: %s", self.server_name, e)
            await self.disconnect()
            return False

    async def disconnect(self):
        """断开连接。"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
        except Exception:
            pass
        try:
            if self._context:
                await self._context.__aexit__(None, None, None)
                self._context = None
        except Exception:
            pass
        self.tools = []

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """调用 MCP 工具。"""
        if not self.session:
            return {"error": f"未连接到 {self.server_name}"}
        try:
            result = await self.session.call_tool(tool_name, arguments)
            # 提取文本内容
            contents = []
            for item in result.content:
                if hasattr(item, "text"):
                    contents.append(item.text)
                elif hasattr(item, "type"):
                    contents.append(f"[{item.type}]")
            return {"content": "\n".join(contents), "is_error": result.isError}
        except Exception as e:
            return {"error": str(e)}

    @property
    def is_connected(self) -> bool:
        return self.session is not None


class MCPConnectionManager:
    """MCP 连接管理器，维护与多个 MCP Server 的连接。"""

    def __init__(self):
        self._connections: dict[str, MCPConnection] = {}

    async def connect(self, server_name: str, url: str, transport: str = "sse", args: list[str] = None) -> bool:
        """连接到 MCP Server。

        Args:
            server_name: Server 名称
            url: SSE URL 或 stdio 命令路径
            transport: 'sse' 或 'stdio'
            args: stdio 模式下的命令参数
        """
        if server_name in self._connections:
            await self._connections[server_name].disconnect()

        conn = MCPConnection(server_name, url, transport)
        success = await conn.connect(args=args)
        if success:
            self._connections[server_name] = conn
        return success

    async def disconnect(self, server_name: str):
        """断开指定 Server。"""
        conn = self._connections.pop(server_name, None)
        if conn:
            await conn.disconnect()

    async def disconnect_all(self):
        """断开所有连接。"""
        for name in list(self._connections.keys()):
            await self.disconnect(name)

    def get_available_tools(self, server_name: str) -> list[dict]:
        """获取指定 Server 的工具列表。"""
        conn = self._connections.get(server_name)
        return conn.tools if conn else []

    def get_all_tools(self) -> dict[str, list[dict]]:
        """获取所有已连接 Server 的工具。"""
        return {name: conn.tools for name, conn in self._connections.items() if conn.is_connected}

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict,
                        tenant_id: int = 0, user_id: int = 0) -> dict:
        """调用 MCP 工具。"""
        conn = self._connections.get(server_name)
        if not conn:
            return {"error": f"未找到 Server: {server_name}"}
        result = await conn.call_tool(tool_name, arguments)

        from backend.billing.audit import log_audit_event
        from backend.storage.database import SessionLocal
        _db = SessionLocal()
        try:
            risk_level = "high" if any(kw in tool_name.lower() for kw in ["write", "delete", "send", "execute"]) else "low"
            log_audit_event(
                db=_db, tenant_id=tenant_id, user_id=user_id,
                action="mcp_tool_call", target=f"{server_name}/{tool_name}",
                arguments=str(arguments)[:2000],
                result_summary=str(result.get("content", ""))[:500],
                risk_level=risk_level,
            )
        finally:
            _db.close()

        return result

    def is_connected(self, server_name: str) -> bool:
        """检查连接状态。"""
        conn = self._connections.get(server_name)
        return conn.is_connected if conn else False

    def list_servers(self) -> list[str]:
        """列出所有已注册的 Server 名称。"""
        return list(self._connections.keys())


# 全局单例
_mcp_manager: MCPConnectionManager | None = None


def get_mcp_manager() -> MCPConnectionManager:
    """获取全局 MCP 连接管理器单例。"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPConnectionManager()
    return _mcp_manager
