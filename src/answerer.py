import os
os.environ["HF_HOME"] = "D:\\hf_cache"

from groq import Groq
from src.retriever import search
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful code assistant.
You are given code snippets from a GitHub repository.
Answer the user's question using ONLY the provided code.
Always mention the file name and line numbers when referring to specific code.
If the answer is not in the provided code, say so honestly."""

def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Snippet {i}] {c['filepath']} "
            f"(lines {c['start_line']}–{c['end_line']})\n"
            f"```\n{c['text'][:1500]}\n```"
        )
    return "\n\n".join(parts)

def ask(question: str) -> tuple[str, list[dict]]:
    print(f"Searching for: {question}")
    chunks = search(question, top_k=6)
    if not chunks:
        return "No relevant code found for your question.", []
    context = format_context(chunks)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Code context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    answer = response.choices[0].message.content
    return answer, chunks