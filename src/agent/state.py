"""
State schema for the Agentic AI Retention Strategy Assistant.
Defines the shared state passed between LangGraph nodes.
"""
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages


class CustomerProfile(TypedDict, total=False):
    """Structured customer information from ML prediction."""
    gender: str
    senior_citizen: bool
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    churn_probability: float
    risk_level: str  # HIGH / MEDIUM / LOW


class AgentState(TypedDict, total=False):
    """The state that flows through the LangGraph retention workflow."""
    # Conversation
    messages: Annotated[list, add_messages]

    # Customer context
    customer_profile: Optional[CustomerProfile]

    # Agent reasoning outputs
    risk_summary: Optional[str]
    recommendations: Optional[str]
    sources: Optional[List[str]]
    disclaimer: Optional[str]

    # Control flow
    needs_rag: Optional[bool]
    session_id: Optional[str]
