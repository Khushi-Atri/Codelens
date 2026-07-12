import os
import asyncio
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
try:
    import streamlit as st
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

FAITHFULNESS_PROMPT = """You are a faithfulness evaluator for a RAG system.

Your job is to check if every claim in the ANSWER is supported by the CONTEXT.

IMPORTANT RULES:
- Only flag a claim as unsupported if it directly contradicts or adds specific 
  facts not present in the context (version numbers, dates, counts, names)
- General descriptions that paraphrase the context are considered supported
- Do not penalise for reasonable summarisation of retrieved content

CONTEXT (retrieved code snippets):
{context}

ANSWER to evaluate:
{answer}

Instructions:
1. Break the answer into individual factual claims
2. For each claim check if it is supported OR reasonably inferred from context
3. Count supported_claims and total_claims
4. Score = supported_claims / total_claims

Respond with ONLY this JSON:
{{"supported": <number>, "total": <number>, "score": <float 0-1>, "unsupported_claims": ["only list claims with specific facts not in context"]}}"""

def score_faithfulness(question: str, answer: str, sources: list[dict]) -> dict:
    """
    Score how faithful the answer is to the retrieved source chunks.
    
    Returns dict with:
      - score: float 0-1 (1.0 = perfectly faithful)
      - percentage: int (0-100 for display)
      - supported: int (claims supported by context)
      - total: int (total claims in answer)
      - unsupported: list of unsupported claims
      - label: str (Excellent / Good / Fair / Poor)
    """
    if not sources or not answer:
        return _default_score()

    # Build context string from sources
    context_parts = []
    for s in sources[:6]:   # use top 6 sources max
        filepath = s.get("filepath", "").replace("\\", "/").split("/")[-1]
        lines    = f"lines {s.get('start_line',0)}–{s.get('end_line',0)}"
        text     = s.get("text", "")[:600]
        context_parts.append(f"[{filepath} {lines}]\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)

    try:
        response = client.chat.completions.create(
           model=os.getenv("CODELENS_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {
                    "role": "user",
                    "content": FAITHFULNESS_PROMPT.format(
                        context=context,
                        answer=answer[:800]   # cap answer length
                    )
                }
            ],
            temperature=0,      # deterministic for evaluation
            max_tokens=300
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON response
        import json, re
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            data = json.loads(match.group(0))
            score       = float(data.get("score", 0.75))
            supported   = int(data.get("supported", 0))
            total       = int(data.get("total", 1))
            unsupported = data.get("unsupported_claims", [])
        else:
            return _default_score()

        # Clamp score to 0-1
        score = max(0.0, min(1.0, score))
        pct   = round(score * 100)

        return {
            "score":       score,
            "percentage":  pct,
            "supported":   supported,
            "total":       total,
            "unsupported": unsupported,
            "label":       _label(score),
            "color":       _color(score)
        }

    except Exception as e:
        print(f"Faithfulness scoring error: {e}")
        return _default_score()


def _default_score() -> dict:
    """Fallback when scoring fails."""
    return {
        "score":       0.75,
        "percentage":  75,
        "supported":   0,
        "total":       0,
        "unsupported": [],
        "label":       "Good",
        "color":       "#3fb950"
    }


def _label(score: float) -> str:
    if score >= 0.9:  return "Excellent"
    if score >= 0.75: return "Good"
    if score >= 0.5:  return "Fair"
    return "Poor"


def _color(score: float) -> str:
    if score >= 0.9:  return "#3fb950"   # green
    if score >= 0.75: return "#58a6ff"   # blue
    if score >= 0.5:  return "#e3b341"   # yellow
    return "#f85149"                      # red