"""
retriever.py — Disease information retrieval coordinator.

Retrieval priority chain:
    1. Curated knowledge base (knowledge_base.py)  — always available for 28 known classes
    2. ChromaDB semantic search                    — for fuzzy/general queries
    3. Wikipedia REST API                          — supplementary background
    4. Groq LLM fallback                          — last resort

Note: The LangChain tools in tools.py use this priority chain directly.
      This module provides a standalone retriever class for external use
      (e.g., evaluation scripts, batch retrieval, future API endpoints).
"""

from .knowledge_base import lookup_disease, semantic_search, format_profile


class DiseaseRetriever:
    """
    Standalone retriever for plant disease information.
    Uses the curated knowledge base as the primary source.
    """

    def fetch_info(self, disease_name: str, confidence_score: float = 1.0) -> str:
        """
        Retrieve structured disease information.

        Args:
            disease_name:     Disease label (as returned by PlantClassifier)
            confidence_score: Classifier confidence (0.0–1.0)

        Returns:
            Formatted disease profile string, or a low-confidence warning.
        """
        if confidence_score < 0.3:
            return (
                "Confidence is too low (< 30%) to retrieve accurate disease information. "
                "Please provide a clearer, closer photograph of the affected leaf in natural light."
            )

        # Primary: exact curated KB lookup
        profile = lookup_disease(disease_name)
        if profile:
            return format_profile(profile)

        # Secondary: semantic search
        results = semantic_search(disease_name, n_results=2)
        if results:
            return "\n\n---\n\n".join(results)

        return f"No information found for '{disease_name}' in the knowledge base."
