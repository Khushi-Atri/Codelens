import os
import pathlib
import pickle
import json
import re
from src.retriever import search as vector_search

# ── paths ────────────────────────────────────────────────────────────
CHUNKS_PATH  = "data/chunks.pkl"
REPO_PATH    = "repo_indexed"

# ── helper: load chunks once ─────────────────────────────────────────
_chunks_cache = None

def _load_chunks() -> list[dict]:
    """Load chunks from disk, cached in memory."""
    global _chunks_cache
    if _chunks_cache is None:
        if not os.path.exists(CHUNKS_PATH):
            return []
        with open(CHUNKS_PATH, "rb") as f:
            _chunks_cache = pickle.load(f)
    return _chunks_cache

def clear_cache():
    """Call this after re-indexing a new repo."""
    global _chunks_cache
    _chunks_cache = None


# ════════════════════════════════════════════════════════════════════
# TOOL 1 — search_code
# The main hybrid retrieval tool the agent uses most often.
# ════════════════════════════════════════════════════════════════════

def search_code(query: str, top_k: int = 6) -> dict:
    """
    Search the indexed codebase using hybrid BM25 + vector retrieval.

    Use this when:
    - You need to find where something is implemented
    - You want to understand how a concept works
    - You need to find functions/classes related to a topic

    Args:
        query:  plain English description of what you're looking for
        top_k:  number of results to return (default 6)

    Returns:
        dict with 'results' list and 'summary' string
    """
    try:
        results = vector_search(query, top_k=top_k)

        if not results:
            return {
                "tool": "search_code",
                "query": query,
                "results": [],
                "summary": f"No results found for '{query}'."
            }

        formatted = []
        for r in results:
            formatted.append({
                "filepath":   r["filepath"],
                "name":       r.get("name", "unknown"),
                "type":       r.get("type", "unknown"),
                "start_line": r["start_line"],
                "end_line":   r["end_line"],
                "preview":    r["text"][:400]   # first 400 chars as preview
            })

        summary_lines = [f"Found {len(results)} results for '{query}':"]
        for i, r in enumerate(formatted, 1):
            fp = r["filepath"].replace("\\", "/")
            summary_lines.append(
                f"  {i}. {r['name']} ({r['type']}) in {fp} "
                f"[lines {r['start_line']}–{r['end_line']}]"
            )

        return {
            "tool":    "search_code",
            "query":   query,
            "results": formatted,
            "summary": "\n".join(summary_lines)
        }

    except Exception as e:
        return {
            "tool":    "search_code",
            "query":   query,
            "results": [],
            "summary": f"Search failed: {str(e)}"
        }


# ════════════════════════════════════════════════════════════════════
# TOOL 2 — get_file
# Read the full contents of any file in the indexed repo.
# ════════════════════════════════════════════════════════════════════

def get_file(filepath: str, max_lines: int = 150) -> dict:
    """
    Read the contents of a specific file from the repository.

    Use this when:
    - You found a file in search_code results and want to read more of it
    - You want to understand the full structure of a specific file
    - You need to see imports, class definitions, or module-level code

    Args:
        filepath:  relative path to the file (e.g. 'fastapi/routing.py')
        max_lines: max lines to return to avoid overflowing context (default 150)

    Returns:
        dict with 'content', 'total_lines', 'filepath'
    """
    # Try to resolve the path relative to the repo folder
    candidates = [
        pathlib.Path(filepath),
        pathlib.Path(REPO_PATH) / filepath,
        pathlib.Path(filepath.lstrip("/\\")),
    ]

    # Also search chunks for the filepath to get the real disk path
    chunks = _load_chunks()
    for chunk in chunks:
        if filepath in chunk["filepath"] or chunk["filepath"].endswith(filepath):
            candidates.insert(0, pathlib.Path(chunk["filepath"]))
            break

    resolved = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            resolved = candidate
            break

    if resolved is None:
        return {
            "tool":     "get_file",
            "filepath": filepath,
            "content":  None,
            "summary":  f"File not found: '{filepath}'. Use list_files() to find the correct path."
        }

    try:
        lines   = resolved.read_text(errors="ignore").splitlines()
        total   = len(lines)
        preview = lines[:max_lines]
        content = "\n".join(preview)
        truncated = total > max_lines

        return {
            "tool":        "get_file",
            "filepath":    str(resolved),
            "content":     content,
            "total_lines": total,
            "shown_lines": len(preview),
            "truncated":   truncated,
            "summary": (
                f"File: {resolved} ({total} lines total, showing first {len(preview)})\n"
                + ("⚠ truncated — use get_function() for specific functions\n" if truncated else "")
                + f"\n{content}"
            )
        }

    except Exception as e:
        return {
            "tool":     "get_file",
            "filepath": filepath,
            "content":  None,
            "summary":  f"Could not read file: {str(e)}"
        }


# ════════════════════════════════════════════════════════════════════
# TOOL 3 — list_files
# Find files in the repo matching a name or pattern.
# ════════════════════════════════════════════════════════════════════

def list_files(pattern: str = "", extension: str = "") -> dict:
    """
    List files in the indexed repository matching a name or extension.

    Use this when:
    - You want to see what files exist in a particular area
    - You're looking for a file by name but don't know the full path
    - You want to understand the project structure

    Args:
        pattern:    substring to match in filename (e.g. 'auth', 'model', 'test')
                    leave empty to list all files
        extension:  file extension to filter by (e.g. '.py', '.js')
                    leave empty for all extensions

    Returns:
        dict with 'files' list and 'summary' string
    """
    chunks = _load_chunks()

    if not chunks:
        return {
            "tool":    "list_files",
            "pattern": pattern,
            "files":   [],
            "summary": "No files indexed yet."
        }

    # Collect unique filepaths from chunks
    seen    = set()
    matches = []

    for chunk in chunks:
        fp = chunk["filepath"]
        if fp in seen:
            continue
        seen.add(fp)

        fp_norm  = fp.replace("\\", "/").lower()
        pat_low  = pattern.lower()
        ext_low  = extension.lower()

        name_match = (pat_low == "") or (pat_low in fp_norm.split("/")[-1])
        ext_match  = (ext_low == "") or fp_norm.endswith(ext_low)

        if name_match and ext_match:
            matches.append(fp.replace("\\", "/"))

    matches.sort()

    # Build a summary showing directory structure
    summary_lines = [f"Found {len(matches)} files matching pattern='{pattern}' ext='{extension}':"]
    for fp in matches[:40]:   # cap at 40 to avoid context overflow
        summary_lines.append(f"  {fp}")
    if len(matches) > 40:
        summary_lines.append(f"  ... and {len(matches) - 40} more")

    return {
        "tool":    "list_files",
        "pattern": pattern,
        "files":   matches,
        "count":   len(matches),
        "summary": "\n".join(summary_lines)
    }


# ════════════════════════════════════════════════════════════════════
# TOOL 4 — get_function
# Extract a specific function or class from the indexed chunks.
# ════════════════════════════════════════════════════════════════════

def get_function(name: str, filepath: str = "") -> dict:
    """
    Extract the full source code of a specific function or class by name.

    Use this when:
    - You found a function name in search results and want its full code
    - You want to read a specific class definition completely
    - You need to understand exactly what a named function does

    Args:
        name:      function or class name to look for (e.g. 'include_router')
        filepath:  optional — narrow search to a specific file

    Returns:
        dict with 'matches' list (may be multiple if name appears in several files)
    """
    chunks  = _load_chunks()
    matches = []

    name_lower = name.lower()
    fp_lower   = filepath.lower().replace("\\", "/")

    for chunk in chunks:
        chunk_name = chunk.get("name", "").lower()
        chunk_fp   = chunk["filepath"].replace("\\", "/").lower()

        # Match by chunk name
        name_match = (chunk_name == name_lower) or (name_lower in chunk_name)

        # Match by filepath if provided
        fp_match = (fp_lower == "") or (fp_lower in chunk_fp) or (chunk_fp.endswith(fp_lower))

        if name_match and fp_match:
            matches.append({
                "name":       chunk.get("name", "unknown"),
                "type":       chunk.get("type", "unknown"),
                "filepath":   chunk["filepath"].replace("\\", "/"),
                "start_line": chunk["start_line"],
                "end_line":   chunk["end_line"],
                "code":       chunk["text"]
            })

    if not matches:
        # Fallback: search by regex in chunk text
        pattern = re.compile(
            rf'\b(def|class)\s+{re.escape(name)}\b',
            re.IGNORECASE
        )
        for chunk in chunks:
            if pattern.search(chunk["text"]):
                chunk_fp = chunk["filepath"].replace("\\", "/")
                fp_match = (fp_lower == "") or (fp_lower in chunk_fp.lower())
                if fp_match:
                    matches.append({
                        "name":       chunk.get("name", name),
                        "type":       chunk.get("type", "unknown"),
                        "filepath":   chunk_fp,
                        "start_line": chunk["start_line"],
                        "end_line":   chunk["end_line"],
                        "code":       chunk["text"]
                    })

    if not matches:
        return {
            "tool":    "get_function",
            "name":    name,
            "matches": [],
            "summary": (
                f"No function or class named '{name}' found"
                + (f" in '{filepath}'" if filepath else "")
                + ". Try search_code() with a description instead."
            )
        }

    summary_lines = [f"Found {len(matches)} match(es) for '{name}':"]
    for i, m in enumerate(matches, 1):
        summary_lines.append(
            f"\n[Match {i}] {m['name']} ({m['type']}) "
            f"in {m['filepath']} lines {m['start_line']}–{m['end_line']}:\n"
            f"```\n{m['code'][:600]}\n```"
        )

    return {
        "tool":    "get_function",
        "name":    name,
        "matches": matches,
        "count":   len(matches),
        "summary": "\n".join(summary_lines)
    }


# ════════════════════════════════════════════════════════════════════
# TOOL REGISTRY — the agent uses this to know what tools exist
# ════════════════════════════════════════════════════════════════════

TOOLS = {
    "search_code": {
        "fn":          search_code,
        "description": "Search the codebase using a natural language query. Returns relevant functions/classes with file paths and line numbers.",
        "args": {
            "query": "string — what you are looking for",
            "top_k": "int (optional) — number of results, default 6"
        }
    },
    "get_file": {
        "fn":          get_file,
        "description": "Read the contents of a specific file. Use when you need to see the full file structure, imports, or module-level code.",
        "args": {
            "filepath":  "string — relative path to the file",
            "max_lines": "int (optional) — max lines to read, default 150"
        }
    },
    "list_files": {
        "fn":          list_files,
        "description": "List files in the repository matching a name pattern or extension. Use to explore project structure.",
        "args": {
            "pattern":   "string (optional) — substring to match in filename",
            "extension": "string (optional) — file extension like '.py' or '.js'"
        }
    },
    "get_function": {
        "fn":          get_function,
        "description": "Get the full source code of a specific function or class by name. Use after finding a name in search results.",
        "args": {
            "name":     "string — function or class name",
            "filepath": "string (optional) — narrow to a specific file"
        }
    }
}


def call_tool(tool_name: str, args: dict) -> dict:
    """
    Call a tool by name with the given arguments.
    Used by the agent loop to execute tool calls.
    """
    if tool_name not in TOOLS:
        return {
            "tool":    tool_name,
            "summary": f"Unknown tool '{tool_name}'. Available: {list(TOOLS.keys())}"
        }

    tool_fn = TOOLS[tool_name]["fn"]

    try:
        return tool_fn(**args)
    except TypeError as e:
        return {
            "tool":    tool_name,
            "summary": f"Wrong arguments for '{tool_name}': {str(e)}"
        }


def get_tools_description() -> str:
    """
    Returns a formatted description of all tools for the LLM system prompt.
    """
    lines = ["You have access to these tools:\n"]
    for name, meta in TOOLS.items():
        lines.append(f"## {name}")
        lines.append(f"Description: {meta['description']}")
        lines.append("Arguments:")
        for arg, desc in meta["args"].items():
            lines.append(f"  - {arg}: {desc}")
        lines.append("")
    return "\n".join(lines)


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Testing agent tools ===\n")

    print("--- list_files('.py') ---")
    r = list_files(extension=".py")
    print(r["summary"][:300])

    print("\n--- search_code('routing') ---")
    r = search_code("how does routing work")
    print(r["summary"])

    print("\n--- get_function('include_router') ---")
    r = get_function("include_router")
    print(r["summary"][:400])