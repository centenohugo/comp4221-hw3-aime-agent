"""Minimal search agent with replaceable model and tools."""
from .agent import Agent, AgentResult
from .tools import Tool, ToolRegistry

__all__ = ["Agent", "AgentResult", "Tool", "ToolRegistry"]
