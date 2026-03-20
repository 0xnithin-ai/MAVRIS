"""
tools.py — LangChain Tool definitions.

Search backend: Wikipedia REST API (pure requests, no native DLLs).
Fallback:       Groq LLM knowledge (when Wikipedia returns nothing).

Flow when user uploads image:
    1. classify_plant_image   -> ViT model -> top class + confidence
    2. fetch_disease_info     -> Wikipedia summary -> if miss, Groq fallback
    3. answer_plant_question  -> Wikipedia + Groq for general questions
"""

import json
import os
import requests
from langchain.tools import tool
from langchain_groq import ChatGroq

from .model import PlantClassifier

# ---------------------------------------------------------------------------
# Shared instances
# ---------------------------------------------------------------------------
_classifier = PlantClassifier()
MODEL_PATH = os.getenv("MODEL_PATH", "plant_model.pth")

_HEADERS = {"User-Agent": "PlantDiseaseAgent/1.0 (educational use)"}
_WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


def _wiki_fetch(query: str) -> str | None:
    """
    Try multiple query variants against Wikipedia REST API.
    Returns the page extract string, or None if nothing found.
    """
    # Build a list of progressively simplified variants to try
    variants = [
        query,
        query.replace(" - ", " "),          # "Potato - Late Blight" -> "Potato Late Blight"
        query.split(" - ")[-1].strip(),     # "Potato - Late Blight" -> "Late Blight"
        query.split("(")[0].strip(),        # strip parentheses suffixes
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
    """Use Groq LLM as a knowledge source when Wikipedia returns nothing."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "No external information available."
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0,
                       max_tokens=600, api_key=api_key)
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
    Returns: top predicted disease, confidence percentage, top-3 predictions, raw class index.
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
        "raw_class_index":  result["raw_index"],
        "top_3_predictions": [
            {"class": cls, "confidence": f"{conf * 100:.1f}%"}
            for cls, conf in result["top3"]
        ],
        "low_confidence_warning": (
            "Confidence is below 40%. The image may be unclear, blurry, or not a plant leaf. "
            "Do not confirm the diagnosis. Ask the user for a better photo."
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
    # Always get Groq expert summary — structured, reliable, covers all sections
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

    # Supplement with Wikipedia background if available
    wiki_info = _wiki_fetch(disease_name)

    if wiki_info:
        return (
            f"BACKGROUND (Wikipedia):\n{wiki_info[:400]}\n\n"
            f"EXPERT KNOWLEDGE:\n{groq_info}"
        )
    return f"EXPERT KNOWLEDGE:\n{groq_info}"



# ---------------------------------------------------------------------------
# Tool 3: Answer general plant care questions
# ---------------------------------------------------------------------------
@tool
def answer_plant_question(question: str) -> str:
    """
    Answer a general plant health or care question that does not require image classification.
    Examples: watering schedules, fertilisation, pest prevention, soil conditions, plant growth.
    Use this tool when no image is provided, or when the user asks a follow-up text question.
    Returns a factual answer from Wikipedia or expert knowledge.
    """
    wiki_result = _wiki_fetch(question)
    if wiki_result:
        return f"[Source: Wikipedia]\n{wiki_result}"

    prompt = (
        f"You are an expert agricultural advisor. Answer this plant care question concisely and "
        f"practically: '{question}'. Do not use emojis."
    )
    return f"[Source: Expert knowledge]\n{_groq_fallback(prompt)}"


TOOLS = [classify_plant_image, fetch_disease_information, answer_plant_question]
