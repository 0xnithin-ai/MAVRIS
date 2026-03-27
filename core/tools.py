"""
tools.py — LangChain Tool definitions.

Retrieval priority chain (researched-grade, reliable):
    PRIMARY:   Curated knowledge base (knowledge_base.py) — always hits for 28 known classes
    SECONDARY: ChromaDB semantic search — for general/fuzzy questions
    TERTIARY:  Wikipedia REST API — supplementary background info
    FALLBACK:  Groq LLM knowledge — last resort

Flow when user uploads image:
    1. classify_plant_image   -> ViT model -> top class + confidence
    2. fetch_disease_info     -> Curated KB (primary) + Wikipedia (supplement)
    3. answer_plant_question  -> ChromaDB KB + Wikipedia + Groq fallback
"""

import json
import os
import requests
import base64
from langchain.tools import tool
from langchain_groq import ChatGroq

from .model import PlantClassifier
from .knowledge_base import lookup_disease, semantic_search, format_profile

# ---------------------------------------------------------------------------
# Shared instances
# ---------------------------------------------------------------------------
_classifier = PlantClassifier()
MODEL_PATH = os.getenv("MODEL_PATH", "plant_model.pth")

_HEADERS = {"User-Agent": "PlantDiseaseAgent/1.0 (educational use)"}
_WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


def _wiki_fetch(query: str) -> "str | None":
    """
    Try multiple query variants against Wikipedia REST API.
    Returns the page extract string, or None if nothing found.
    Used as a supplementary source — not primary.
    """
    variants = [
        query,
        query.replace(" - ", " "),
        query.split(" - ")[-1].strip(),
        query.split("(")[0].strip(),
    ]
    seen = set()
    for v in variants:
        if v in seen or not v:
            continue
        seen.add(v)
        url = _WIKIPEDIA_API.format(requests.utils.quote(v.replace(" ", "_")))
        try:
            r = requests.get(url, headers=_HEADERS, timeout=8)
            if r.status_code == 200:
                data = r.json()
                extract = data.get("extract", "")
                if extract and len(extract) > 80:
                    return extract
        except Exception:
            pass
    return None


def _groq_fallback(prompt: str) -> str:
    """Use Groq LLM as a last-resort knowledge source."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "No external information available."
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,
                       max_tokens=800, api_key=api_key)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Fallback knowledge unavailable: {e}"


# ---------------------------------------------------------------------------
# Tool 1: Classify plant image using the trained ViT model
# ---------------------------------------------------------------------------
@tool
def classify_plant_image(image_path: str) -> str:
    """
    Classify a plant leaf image using the trained ViT model
    (timm vit_base_patch16_224, 28 plant disease classes, weights from plant_model.pth).
    Input: absolute path to a local image file.
    Returns: top predicted disease, calibrated confidence percentage, top-3 predictions, raw class index.
    Always call this tool first when the user uploads an image.
    """
    if not os.path.exists(image_path):
        return json.dumps({"error": f"Image not found at path: {image_path}"})

    result = _classifier.predict(image_path, model_path=MODEL_PATH)
    confidence_pct = result["confidence"] * 100

    return json.dumps({
        "top_prediction":   result["top_class"],
        "confidence":       f"{confidence_pct:.1f}%",
        "confidence_raw":   result["confidence"],
        "calibrated":       result.get("calibrated", False),
        "raw_class_index":  result["raw_index"],
        "top_3_predictions": [
            {"class": cls, "confidence": f"{conf * 100:.1f}%"}
            for cls, conf in result["top3"]
        ],
        "low_confidence_warning": (
            "Confidence is below 40%. Execute TOP-2 prediction Visual Verification loop."
        ) if confidence_pct < 40 else "",
    })


# ---------------------------------------------------------------------------
# Tool 2: Fetch factual disease information
# ---------------------------------------------------------------------------
@tool
def fetch_disease_information(disease_name: str) -> str:
    """
    Retrieve factual information about a specific plant disease: symptoms, causes, treatment, prevention.
    Input: disease name exactly as returned by classify_plant_image
           (e.g., "Potato - Late Blight", "Tomato - Early Blight").
    Always call this after classification. Returns a structured expert summary ready for the final response.
    """
    # ── PRIMARY: Curated knowledge base lookup (always reliable for 28 classes) ──
    profile = lookup_disease(disease_name)
    if profile:
        kb_text = format_profile(profile)
        # Optionally supplement with Wikipedia background
        wiki_info = _wiki_fetch(disease_name)
        if wiki_info:
            return (
                f"[Source: Curated Knowledge Base + Wikipedia]\n\n"
                f"BACKGROUND (Wikipedia):\n{wiki_info[:400]}\n\n"
                f"{kb_text}"
            )
        return f"[Source: Curated Knowledge Base]\n\n{kb_text}"

    # ── SECONDARY: ChromaDB semantic search (for edge cases / partial matches) ──
    semantic_results = semantic_search(disease_name, n_results=2)
    if semantic_results:
        semantic_text = "\n\n---\n\n".join(semantic_results)
        return f"[Source: Knowledge Base (semantic match)]\n\n{semantic_text}"

    # ── TERTIARY: Wikipedia ──
    wiki_info = _wiki_fetch(disease_name)

    # ── FALLBACK: Groq expert knowledge ──
    groq_prompt = (
        f"You are a plant pathologist. Write a concise, factual expert summary for the plant disease "
        f"'{disease_name}'. Structure your answer in exactly these four sections:\n\n"
        f"CAUSE: [1-2 sentences on what causes this disease — pathogen name, type]\n\n"
        f"SYMPTOMS: [3 bullet points of visible symptoms on leaves/plant]\n\n"
        f"TREATMENT: [3 numbered steps a farmer can take right now]\n\n"
        f"PREVENTION: [2 bullet points on how to prevent it]\n\n"
        f"Be specific, practical, and factual. Do not use emojis."
    )
    groq_info = _groq_fallback(groq_prompt)

    if wiki_info:
        return (
            f"[Source: Wikipedia + Expert Knowledge]\n\n"
            f"BACKGROUND:\n{wiki_info[:400]}\n\n"
            f"EXPERT KNOWLEDGE:\n{groq_info}"
        )
    return f"[Source: Expert Knowledge]\n\nEXPERT KNOWLEDGE:\n{groq_info}"


# ---------------------------------------------------------------------------
# Tool 3: Answer general plant care questions
# ---------------------------------------------------------------------------
@tool
def answer_plant_question(question: str) -> str:
    """
    Answer a general plant health or care question that does not require image classification.
    Examples: watering schedules, fertilisation, pest prevention, soil conditions, plant growth.
    Use this tool when no image is provided, or when the user asks a follow-up text question.
    Returns a factual answer from the knowledge base, Wikipedia, or expert knowledge.
    """
    # ── PRIMARY: ChromaDB semantic search over curated KB ──
    semantic_results = semantic_search(question, n_results=3)
    if semantic_results:
        combined = "\n\n---\n\n".join(semantic_results[:2])
        # Also check Wikipedia for supplementary info
        wiki_result = _wiki_fetch(question)
        if wiki_result:
            return (
                f"[Source: Knowledge Base + Wikipedia]\n\n"
                f"FROM KNOWLEDGE BASE:\n{combined}\n\n"
                f"ADDITIONAL CONTEXT (Wikipedia):\n{wiki_result[:300]}"
            )
        return f"[Source: Knowledge Base]\n\n{combined}"

    # ── SECONDARY: Wikipedia ──
    wiki_result = _wiki_fetch(question)
    if wiki_result:
        return f"[Source: Wikipedia]\n{wiki_result}"

    # ── FALLBACK: Groq LLM ──
    prompt = (
        f"You are an expert agricultural advisor. Answer this plant care question concisely and "
        f"practically: '{question}'. Do not use emojis."
    )
    return f"[Source: Expert Knowledge]\n{_groq_fallback(prompt)}"


# ---------------------------------------------------------------------------
# Tool 4: Test-Time Visual Verification (Agentic VLM Loop via Ollama/LLaVA)
# ---------------------------------------------------------------------------
@tool
def verify_visual_trait(image_path: str, trait_question: str) -> str:
    """
    Look exclusively for visual symptoms of a disease on the plant leaf image (Test-Time Verification).
    Inputs:
      - image_path: The absolute path to the local image.
      - trait_question: A specific YES/NO question about visual traits (e.g. "Do you see white cottony sporulation?" or "Are there concentric rings?").
    Returns: The VLM's observation.
    Use this if the confidence of the initial classification is low/uncertain, and you need second-stage visual verification.
    """
    if not os.path.exists(image_path):
        return f"[VLM Verification Failed] Image not found at {image_path}"

    try:
        with open(image_path, "rb") as image_file:
            img_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        # We enforce a concise YES/NO + explanation format from LLaVA
        system_instruction = (
            "You are an expert plant pathologist's vision assistant. Act as an exact, uncreative visual sensor. "
            "Examine this leaf. Answer the question starting strictly with 'YES' or 'NO', followed by a 1 sentence explanation of what you see."
        )
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llava",
            "prompt": f"{system_instruction}\n\nQuestion: {trait_question}",
            "images": [img_b64],
            "stream": False
        }
        
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return f"[Visual Verification Result]\n{data.get('response', 'No response')}"
        
    except Exception as e:
        return f"[VLM Verification Failed] Local LLaVA connection issue. Error: {str(e)}"

TOOLS = [classify_plant_image, fetch_disease_information, answer_plant_question, verify_visual_trait]
