"""
LangGraph workflow for the Agentic AI Retention Strategy Assistant.
Nodes: analyze_risk → retrieve_strategies → generate_plan → respond
"""

import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# ── LLM Setup ──────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    api_key=os.getenv("GROQ_API_KEY"),
)

SYSTEM_PROMPT = """You are an expert AI Customer Retention Strategist for a telecom company.
Your job is to help customer service representatives retain customers who are predicted to churn by our specialized Machine Learning models.

**Core Instruction:**
1. **Trust the ML Prediction:** Always respect the `Risk Level` (HIGH/MEDIUM/LOW) provided in the customer profile. In this industry, a churn probability above 20% is often considered HIGH RISK due to class imbalance. 
2. **Analyze, Don't Re-predict:** Your goal is to explain *why* the ML model reached its conclusion (based on factors like contract type, tenure, and internet service) and provide solutions. Do not downplay a risk simply because the probability score seems low numerically (e.g., 25% is a serious threat).
3. **Be Actionable:** Recommend specific, data-driven retention strategies (discounts, contract upgrades).

Always be professional and reference the specific customer data. Format responses with clear sections and bullet points."""


# ── Node Functions ─────────────────────────────────────────────────────────


def analyze_risk(state: AgentState) -> dict:
    """Analyze customer profile and identify churn drivers."""
    profile = state.get("customer_profile")
    if not profile:
        return {"risk_summary": None, "needs_rag": False}

    analysis_prompt = f"""Analyze this telecom customer's churn risk profile using the data and the Machine Learning prediction provided.

**ML Prediction Result:**
- Churn Probability: {profile.get('churn_probability', 'N/A')}
- Risk Level: {profile.get('risk_level', 'N/A')}

**Customer Data:**
- Tenure: {profile.get('tenure', 'N/A')} months
- Contract: {profile.get('contract', 'N/A')}
- Internet Service: {profile.get('internet_service', 'N/A')}
- Monthly Charges: ${profile.get('monthly_charges', 'N/A')}
- Tech Support: {profile.get('tech_support', 'N/A')}
- Payment Method: {profile.get('payment_method', 'N/A')}

**Instructions:**
1. Provide a concise risk summary that aligns **exactly** with the ML risk level ({profile.get('risk_level', 'N/A')}).
2. Identify the top 3 features (from the data above) that likely influenced the ML model's prediction.
3. Keep the summary under 150 words."""

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=analysis_prompt),
        ]
    )

    return {
        "risk_summary": response.content,
        "needs_rag": True,
    }


def retrieve_strategies(state: AgentState) -> dict:
    """Retrieve relevant retention strategies from the RAG knowledge base."""
    if not state.get("needs_rag"):
        return {"sources": []}

    try:
        from rag.vector_store import get_retriever

        retriever = get_retriever()

        risk_summary = state.get("risk_summary", "")
        profile = state.get("customer_profile", {})

        # Build a targeted query from the customer's specifics
        query_parts = []
        contract = profile.get("contract", "")
        if "Month-to-month" in contract:
            query_parts.append("month-to-month contract retention")
        if profile.get("monthly_charges", 0) > 70:
            query_parts.append("high monthly charges discount")
        if profile.get("internet_service") == "Fiber optic":
            query_parts.append("fiber optic customer retention")
        if profile.get("tenure", 0) < 12:
            query_parts.append("new customer retention short tenure")
        if not query_parts:
            query_parts.append("general customer retention strategies")

        query = " ".join(query_parts)
        docs = retriever.invoke(query)
        sources = [doc.page_content for doc in docs[:3]]
    except Exception as e:
        logger.error(f"⚠️ RAG retrieval failed: {e}")
        sources = [
            "Offer a 15-20% loyalty discount on a 1-year contract upgrade.",
            "Bundle internet security and tech support at a reduced rate.",
            "Provide a free month of premium services as a goodwill gesture.",
            "Assign a dedicated account manager for high-value customers.",
            "Offer a contract buyout incentive to move from month-to-month to annual.",
        ]

    return {"sources": sources}


def generate_retention_plan(state: AgentState) -> dict:
    """Generate structured retention recommendations using chain-of-thought."""
    profile = state.get("customer_profile", {})
    risk_summary = state.get("risk_summary", "No risk summary available.")
    sources = state.get("sources", [])

    sources_text = (
        "\n".join(f"- {s}" for s in sources)
        if sources
        else "No specific strategies retrieved."
    )

    plan_prompt = f"""Based on this customer's risk analysis and our company retention strategies, create a structured retention action plan.

## Risk Summary
{risk_summary}

## Company Retention Strategies & Best Practices
{sources_text}

## Customer Details
- Contract: {profile.get('contract', 'N/A')}
- Monthly Charges: ${profile.get('monthly_charges', 'N/A')}
- Tenure: {profile.get('tenure', 'N/A')} months
- Risk Level: {profile.get('risk_level', 'N/A')}

## Instructions
Think step by step:
1. First, identify the most impactful intervention for this specific customer.
2. Then, create a prioritized list of retention actions.
3. Finally, draft a sample script the CS representative can use.

Create a structured retention plan with:
- **Priority Actions** (numbered, with expected impact)
- **Recommended Offers** (specific discounts/upgrades with amounts)
- **Sample CS Script** (what the rep should say to the customer)
- **Follow-up Plan**

Be extremely concise. Total response must be under 250 words. Focus on actionable steps over explanation. """

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=plan_prompt),
        ]
    )

    disclaimer = (
        "⚖️ **Disclaimer:** These recommendations are generated by an AI model based on "
        "statistical patterns in historical customer data. They should be used as guidelines "
        "and not as definitive business decisions. Individual customer circumstances may vary. "
        "All offers must be approved by management before being extended to customers. "
        "Customer data is handled in compliance with applicable privacy regulations."
    )

    return {
        "recommendations": response.content,
        "disclaimer": disclaimer,
    }


def respond_to_user(state: AgentState) -> dict:
    """Generate the final conversational response."""
    messages = state.get("messages", [])
    risk_summary = state.get("risk_summary")
    recommendations = state.get("recommendations")
    profile = state.get("customer_profile")
    disclaimer = state.get("disclaimer")

    # Build context for the LLM
    context_parts = []
    if profile:
        context_parts.append(
            f"Customer Risk Level: {profile.get('risk_level', 'Unknown')}"
        )
        context_parts.append(
            f"Churn Probability: {profile.get('churn_probability', 'Unknown')}"
        )
        context_parts.append(f"Contract: {profile.get('contract', 'Unknown')}")
        context_parts.append(
            f"Monthly Charges: ${profile.get('monthly_charges', 'Unknown')}"
        )
    if risk_summary:
        context_parts.append(f"\n## Risk Summary\n{risk_summary}")
    if recommendations:
        context_parts.append(f"\n## Retention Plan\n{recommendations}")
    if disclaimer:
        context_parts.append(f"\n{disclaimer}")

    context = "\n".join(context_parts)

    system_with_context = SYSTEM_PROMPT
    if context:
        system_with_context += f"\n\n--- CURRENT CUSTOMER CONTEXT ---\n{context}"

    # Trim history to stay within token limits (e.g., last 4 messages = 2 turns)
    trimmed_messages = messages[-4:] if len(messages) > 4 else messages

    response = llm.invoke(
        [SystemMessage(content=system_with_context)] + trimmed_messages
    )

    return {"messages": [response]}


# ── Routing Logic ──────────────────────────────────────────────────────────


def should_analyze(state: AgentState) -> str:
    """Determine whether to run full analysis or just chat."""
    if state.get("customer_profile") and not state.get("risk_summary"):
        return "analyze"
    return "respond"


# ── Build the Graph ────────────────────────────────────────────────────────


def build_graph():
    """Construct and compile the LangGraph retention workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_risk)
    workflow.add_node("retrieve", retrieve_strategies)
    workflow.add_node("plan", generate_retention_plan)
    workflow.add_node("respond", respond_to_user)

    # Entry point routes to either analysis pipeline or direct response
    workflow.set_conditional_entry_point(
        should_analyze,
        {"analyze": "analyze", "respond": "respond"},
    )

    # Analysis pipeline: analyze → retrieve → plan → respond
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "plan")
    workflow.add_edge("plan", "respond")
    workflow.add_edge("respond", END)

    # Session memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Singleton graph instance
retention_agent = build_graph()
