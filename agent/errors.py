"""Shared exception types."""


class ToolError(Exception):
    """An expected, user-presentable failure of a tool (service down, not found, bad state...)."""
