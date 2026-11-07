"""Errors shared by local execution and MCP tools."""


class ToolError(Exception):
    pass


class NotFoundError(ToolError):
    pass


class ValidationError(ToolError):
    pass
