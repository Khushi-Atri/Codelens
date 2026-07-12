import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
try:
    import streamlit as st
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

from src.agent_tools import call_tool, get_tools_description, clear_cache

client = Groq(api_key=api_key)

MAX_STEPS = 8


def _get_model() -> str:
    """Returns the best available model, falls back to smaller on rate limit."""
    return os.getenv("CODELENS_MODEL", "llama-3.3-70b-versatile")


# ── System prompt ─────────────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are CodeLens, a codebase explorer agent.

{tools}

## STRICT RULES — you MUST follow these exactly

RULE 1: You MUST call at least 2 tools before giving a final_answer. No exceptions.
RULE 2: Your response must ALWAYS be a single JSON object. Nothing else.
RULE 3: Never answer from memory. Always search the code first.
RULE 4: Do not assume anything — verify by calling tools.
RULE 5: NEVER state facts you did not read in the retrieved code.
RULE 6: If you cannot find something, say "I could not find this in the codebase".
RULE 7: It is better to say "I don't know" than to guess.

## Response format

To call a tool:
{{"action": "tool_name", "args": {{"arg1": "value1"}}, "thought": "reason"}}

To give final answer (only after calling tools):
{{"action": "final_answer", "answer": "your answer", "thought": "I searched and found enough"}}

## Available actions
- search_code: search codebase by meaning
- get_file: read a specific file
- list_files: find files by name pattern
- get_function: extract a specific function

## Example flow for "what does this project do?"

Step 1: {{"action": "list_files", "args": {{"pattern": ""}}, "thought": "Let me see all files first"}}
Step 2: {{"action": "search_code", "args": {{"query": "main application purpose entry point"}}, "thought": "Search for what the app does"}}
Step 3: {{"action": "get_file", "args": {{"filepath": "README.md"}}, "thought": "Read the README for project description"}}
Step 4: {{"action": "final_answer", "answer": "...", "thought": "I have enough information"}}

Remember: ONLY output valid JSON. No text before or after. No markdown.
Do NOT include any text before or after the JSON object.
Do NOT wrap JSON in markdown code blocks.
Your ENTIRE response must be parseable as JSON."""


# ── JSON parser ───────────────────────────────────────────────────────
def parse_llm_response(text: str) -> dict:
    """
    Parse the LLM's JSON response robustly.
    Handles markdown, extra text, and partial JSON.
    """
    text = text.strip()

    # Remove common prefixes the model adds
    for prefix in ["Here is my response:", "My response:", "Response:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try markdown code block ```json ... ```
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find the LAST { ... } block (most complete one)
    matches = list(re.finditer(r'\{[\s\S]*?\}', text))
    for m in reversed(matches):
        try:
            parsed = json.loads(m.group(0))
            if "action" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Try greedy { ... } from first { to last }
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

    # Last resort — treat entire text as final answer
    return {
        "action":  "final_answer",
        "answer":  text,
        "thought": "Direct text response"
    }

def is_codebase_question(question: str) -> bool:
    """Returns False if the question is not about the indexed codebase."""
    out_of_scope = [
        "suggest", "recommend", "best project", "final year",
        "what should i", "give me ideas", "career advice",
        "which language", "teach me", "explain concept",
        "what is the best", "should i learn"
    ]
    q = question.lower()
    return not any(phrase in q for phrase in out_of_scope)

# ── Main agent loop ───────────────────────────────────────────────────
def run_agent(question: str, chat_history: list = None) -> dict:
    # Detect out-of-scope questions early
    if not is_codebase_question(question):
        return {
            "answer": "I can only answer questions about the indexed codebase — things like how specific features work, where functions are defined, what the tech stack is, etc. For general advice or recommendations, I'm not the right tool.",
            "steps":  [],
            "sources": []
        }
    
    
    """
    Run the ReAct agent loop.

    Args:
        question:     the user's question
        chat_history: list of past messages for memory (optional)

    Returns:
        dict with:
          - answer:  the final answer string
          - steps:   list of step dicts (for the thinking panel UI)
          - sources: list of source chunks used
    """
    steps   = []
    sources = []
    messages = []

    # Add past conversation for memory (last 2 messages only)
    if chat_history:
        for msg in chat_history[-2:]:
            content = msg["content"]
            # Skip raw JSON assistant messages — they confuse the model
            if msg["role"] == "assistant" and content.strip().startswith("{"):
                continue
            messages.append({
                "role":    msg["role"],
                "content": content[:500]
            })

    # Add the current question
    messages.append({
        "role":    "user",
        "content": question
    })

    # ── Agent loop ────────────────────────────────────────────────────
    for step_num in range(MAX_STEPS):

        # Ask the LLM what to do next
        try:
            response = client.chat.completions.create(
                model=_get_model(),
                messages=[
                    {
                        "role": "system",
                        "content": AGENT_SYSTEM_PROMPT.format(
                            tools=get_tools_description()
                        )
                    }
                ] + messages,
                temperature=0.1,
                max_tokens=1024
            )
        except Exception as e:
            err = str(e)
            # Rate limit — auto switch to smaller model
            if "rate_limit_exceeded" in err or "429" in err:
                os.environ["CODELENS_MODEL"] = "llama-3.1-8b-instant"
                return {
                    "answer":  "⚠ Rate limit hit on 70B model — switched to smaller model. Please ask your question again.",
                    "steps":   steps,
                    "sources": sources
                }
            return {
                "answer":  f"Agent error calling LLM: {err}",
                "steps":   steps,
                "sources": sources
            }

        # Parse the LLM response
        raw    = response.choices[0].message.content
        parsed = parse_llm_response(raw)

        action  = parsed.get("action", "final_answer")
        thought = parsed.get("thought", "")
        args    = parsed.get("args", {})

        # ── Final answer — done ──
        if action == "final_answer":
            answer = parsed.get("answer", raw)
            steps.append({
                "step":    step_num + 1,
                "type":    "final",
                "thought": thought,
                "result":  answer
            })
            if len(steps) >= 3:
                last_3 = [s.get("tool") for s in steps[-3:] if s["type"] == "tool"]
                if len(set(last_3)) == 1 and len(last_3) == 3:
                  break
            return {
                "answer":  answer,
                "steps":   steps,
                "sources": sources
            }

        # ── Tool call ──
        tool_name = action

        # Record this step
        steps.append({
            "step":    step_num + 1,
            "type":    "tool",
            "tool":    tool_name,
            "args":    args,
            "thought": thought,
            "result":  None
        })

        # Call the tool
        try:
            tool_result = call_tool(tool_name, args)
        except Exception as e:
            tool_result = {
                "tool":    tool_name,
                "summary": f"Tool error: {str(e)}"
            }

        # Collect sources from search and function lookup tools
        if tool_name in ("search_code", "get_function"):
            results = tool_result.get("results") or tool_result.get("matches") or []
            for r in results:
                sources.append({
                    "filepath":   r.get("filepath", ""),
                    "name":       r.get("name", "unknown"),
                    "type":       r.get("type", "unknown"),
                    "start_line": r.get("start_line", 0),
                    "end_line":   r.get("end_line", 0),
                    "text":       r.get("preview") or r.get("code", "")
                })

        # Update step result
        steps[-1]["result"] = tool_result.get("summary", str(tool_result))

        # Add tool call + result to history so LLM remembers it
        messages.append({
            "role":    "assistant",
            "content": raw
        })
        messages.append({
            "role":    "user",
            "content": f"Tool result for {tool_name}:\n{tool_result.get('summary', '')}"
        })

    # ── Max steps reached — get best answer from what was gathered ────
    try:
        final_response = client.chat.completions.create(
            model=_get_model(),
            messages=[
                {
                    "role": "system",
                    "content": AGENT_SYSTEM_PROMPT.format(
                        tools=get_tools_description()
                    )
                }
            ] + messages + [{
                "role":    "user",
                "content": "You have reached the maximum number of tool calls. Give your best final answer now based on everything you retrieved so far. Respond with the final_answer JSON."
            }],
            temperature=0.1,
            max_tokens=1024
        )
        parsed_final = parse_llm_response(final_response.choices[0].message.content)
        answer = parsed_final.get("answer", final_response.choices[0].message.content)
    except Exception as e:
        answer = f"Reached max steps. Error getting final answer: {str(e)}"

    steps.append({
        "step":    MAX_STEPS + 1,
        "type":    "final",
        "thought": "Max steps reached — giving best answer from gathered info",
        "result":  answer
    })

    return {
        "answer":  answer,
        "steps":   steps,
        "sources": sources
    }


# ── Auto codebase summary ─────────────────────────────────────────────
def auto_summarize_repo() -> dict:
    """
    Automatically explores the repo and generates a summary.
    Called right after indexing — no user question needed.
    """
    summary_question = """Explore this codebase and tell me:
1. What does this project do? (1-2 sentences)
2. What is the main entry point?
3. What are the 3-4 core modules or folders?
4. What tech stack or frameworks are used?

Use list_files and search_code to explore before answering."""

    result = run_agent(summary_question)

    return {
        "summary": result["answer"],
        "steps":   result["steps"]
    }