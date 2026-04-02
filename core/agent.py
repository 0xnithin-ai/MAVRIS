"""
agent.py — LangChain/LangGraph agent with Groq LLM and tool binding.

Architecture:
    User query + image path
        -> ChatGroq (llama-3.3-70b-versatile)           ← upgraded from 8b-instant
        -> langgraph.prebuilt.create_react_agent  (tool calling loop)
            -> classify_plant_image      (timm ViT-B/16, plant_model.pth, 28 classes)
            -> fetch_disease_information (Curated KB → Wikipedia → Groq fallback)
            -> verify_visual_trait       (LLaVA 7B via Ollama — visual symptom check)
            -> synthesize_final_diagnosis (Groq LLaMA-3.3-70B — LLM Arbitrator/Judge)
        -> Final structured response using synthesis verdict
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from .tools import TOOLS
from .prompts import SYSTEM_PROMPT

# Hard cap: agent cannot call more than this many steps total (prevents infinite loops)
# Budget: classify(1) + fetch(1) + verify(0–2) + synthesize(1) + LLM turns(~6) + headroom
_MAX_ITERATIONS = 22


def build_agent():
    """
    Build and return the LangGraph ReAct agent.
    Called once at startup; result is cached in Streamlit session_state.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",   # upgraded: better reasoning & instruction following
        temperature=0,
        max_tokens=2048,
        api_key=api_key,
    )

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    return agent


class PlantDiseaseAgent:
    """
    Thin wrapper around the LangGraph ReAct agent.
    Injects the image path into the human message so the LLM
    always knows which file to classify.
    """

    def __init__(self):
        self._agent = build_agent()

    def run(self, user_query: str, image_path: str = None) -> str:
        """
        Run the full agent pipeline.

        Args:
            user_query:  Natural language question from the user.
            image_path:  Absolute path to the uploaded image (or None).

        Returns:
            str: Final agent response.
        """
        if image_path and os.path.exists(image_path):
            full_query = (
                f"Image file path: {image_path}\n\n"
                f"User question: {user_query}"
            )
        else:
            full_query = user_query

        # recursion_limit caps the total number of graph steps (tool calls + LLM calls).
        # At _MAX_ITERATIONS=8: supports classify + retrieve + answer with headroom.
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=full_query)]},
            config={"recursion_limit": _MAX_ITERATIONS},
        )

        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return "No response was generated. Please try again."
