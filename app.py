import os
os.environ["HF_HOME"] = "D:\\hf_cache"

import streamlit as st
from src.cloner import clone_repo, collect_files
from src.chunker import chunk_repo
from src.embedder import embed_and_save
from src.agent import run_agent, auto_summarize_repo
from src.agent_tools import clear_cache
from src.evaluator import score_faithfulness

st.set_page_config(
    page_title="CodeLens",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    background-color: #0a0a0f !important;
    color: #c9d1d9 !important;
}

.stApp,
.stApp > header,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHorizontalBlock"],
[data-testid="stVerticalBlock"],
[data-testid="column"],
section.main, .main, .block-container {
    background-color: #0a0a0f !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 1rem !important;
    max-width: 960px !important;
    margin: 0 auto !important;
}

p, span, div, label, li, a { color: #c9d1d9; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }

.topbar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #21262d;
}
.cl-logo { font-size: 18px; font-weight: 600; color: #7c6af7 !important; letter-spacing: -0.5px; text-shadow: 0 0 20px rgba(124,106,247,0.5); }
.cl-logo span { color: #484f58 !important; font-weight: 300; font-size: 13px; margin-left: 6px; }
.cl-badge { font-size: 10px; background: #1c1f3a; border: 1px solid #3d3580; color: #7c6af7 !important; padding: 3px 10px; border-radius: 20px; font-weight: 500; }

.stTextInput > div > div > input {
    background: #161b22 !important; border: 1px solid #30363d !important;
    border-radius: 8px !important; color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;
    padding: 12px 14px !important; transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus { border-color: #7c6af7 !important; box-shadow: 0 0 0 3px rgba(124,106,247,0.15) !important; outline: none !important; }
.stTextInput > div > div > input::placeholder { color: #484f58 !important; }
.stTextInput label { color: #484f58 !important; font-size: 11px !important; }

.stButton > button {
    background: linear-gradient(135deg, #7c6af7, #5a4fcf) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;
    font-weight: 500 !important; padding: 10px 20px !important; cursor: pointer !important;
    transition: all 0.2s !important; box-shadow: 0 0 20px rgba(124,106,247,0.3) !important; width: 100% !important;
}
.stButton > button:hover { box-shadow: 0 0 30px rgba(124,106,247,0.5) !important; transform: translateY(-1px) !important; color: #fff !important; }

.stat-row { display: flex; gap: 10px; margin: 1.2rem 0; }
.stat-card { flex: 1; background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 10px 14px; transition: border-color 0.2s; }
.stat-card:hover { border-color: #7c6af7; }
.stat-num { font-size: 20px; font-weight: 600; color: #7c6af7 !important; }
.stat-label { font-size: 9px; color: #484f58 !important; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.06em; }

.repo-tag { display: inline-flex; align-items: center; gap: 8px; background: #161b22; border: 1px solid #21262d; border-left: 3px solid #7c6af7; border-radius: 6px; padding: 8px 14px; font-size: 12px; margin-bottom: 1rem; }
.repo-tag-name { color: #e6edf3 !important; font-weight: 500; }
.repo-tag-meta { color: #484f58 !important; font-size: 10px; }

.summary-card { background: #0d1117; border: 1px solid #21262d; border-left: 3px solid #3fb950; border-radius: 8px; padding: 14px 16px; margin: 1rem 0; font-size: 12px; line-height: 1.7; color: #c9d1d9 !important; }
.summary-title { font-size: 9px; font-weight: 600; letter-spacing: .1em; color: #3fb950 !important; text-transform: uppercase; margin-bottom: 8px; }

.thinking-panel { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px 14px; margin: 8px 0 12px; font-family: monospace; }
.thinking-title { font-size: 9px; font-weight: 600; letter-spacing: .1em; color: #484f58 !important; text-transform: uppercase; margin-bottom: 10px; }
.think-step { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px; animation: fadeIn 0.3s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateX(-6px); } to { opacity: 1; transform: translateX(0); } }
.think-icon { width: 20px; height: 20px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; flex-shrink: 0; margin-top: 1px; }
.think-icon.tool  { background: #1c1f3a; color: #7c6af7 !important; }
.think-icon.final { background: #0d1f0f; color: #3fb950 !important; }
.think-body { flex: 1; }
.think-tool { font-size: 11px; font-weight: 600; color: #7c6af7 !important; margin-bottom: 2px; }
.think-tool.final { color: #3fb950 !important; }
.think-thought { font-size: 10px; color: #8b949e !important; line-height: 1.5; }
.think-result { font-size: 10px; color: #484f58 !important; margin-top: 3px; font-style: italic; }

.divider { height: 1px; background: #21262d; margin: 1.5rem 0; }

.msg-wrap { margin-bottom: 1.5rem; animation: msgFade 0.3s ease; }
@keyframes msgFade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.msg-role { font-size: 9px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
.msg-role::before { content: ''; display: inline-block; width: 4px; height: 4px; border-radius: 50%; background: currentColor; }
.msg-role.user      { color: #7c6af7 !important; }
.msg-role.assistant { color: #3fb950 !important; }
.msg-bubble         { font-size: 13px; line-height: 1.75; color: #c9d1d9 !important; }
.msg-bubble.user    { color: #e6edf3 !important; }

.chips-row { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 10px; }
.chip { font-size: 10px; background: #161b22; border: 1px solid #21262d; border-radius: 5px; padding: 3px 8px; color: #58a6ff !important; transition: all 0.15s; font-family: monospace; }
.chip:hover { border-color: #58a6ff; background: #1c2333; }

/* ── Faithfulness bar — dynamic color support ── */
.faith-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.faith-lbl { font-size: 9px; color: #484f58 !important; text-transform: uppercase; letter-spacing: .08em; white-space: nowrap; }
.faith-track { flex: 1; height: 3px; background: #21262d; border-radius: 2px; overflow: hidden; min-width: 60px; }
.faith-pct { font-size: 10px; font-weight: 600; white-space: nowrap; }
.faith-label-badge { font-size: 9px; font-weight: 600; padding: 2px 7px; border-radius: 10px; white-space: nowrap; }
.faith-unsupported { font-size: 10px; color: #f85149 !important; margin-top: 5px; width: 100%; }

.step-badge { display: inline-block; font-size: 9px; font-weight: 600; background: #1c1f3a; border: 1px solid #3d3580; color: #7c6af7 !important; padding: 2px 8px; border-radius: 10px; margin-top: 8px; }

.quick-label { font-size: 9px; color: #484f58 !important; margin: 1.2rem 0 0.6rem; text-transform: uppercase; letter-spacing: .08em; }

.stProgress > div > div { background: #7c6af7 !important; }
.stProgress { background: #21262d !important; border-radius: 4px !important; }

[data-testid="stExpander"] { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 6px !important; }
[data-testid="stExpander"] summary { color: #8b949e !important; font-size: 11px !important; }
[data-testid="stExpander"] p, [data-testid="stExpander"] span { color: #c9d1d9 !important; }

.stCodeBlock, [data-testid="stCodeBlock"] { background: #0d1117 !important; border: 1px solid #21262d !important; border-radius: 6px !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: #484f58 !important; }

[data-testid="stChatInput"] textarea { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; border-radius: 8px !important; }
[data-testid="stChatInput"] textarea:focus { border-color: #7c6af7 !important; box-shadow: 0 0 0 3px rgba(124,106,247,0.15) !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #484f58 !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: #7c6af7 !important; }

.stSuccess { background: #0d1f0f !important; border: 1px solid #3fb950 !important; color: #3fb950 !important; font-family: monospace !important; }
.stError   { background: #1f0d0d !important; border: 1px solid #f85149 !important; color: #f85149 !important; font-family: monospace !important; }
[data-testid="stSpinner"] p { color: #8b949e !important; }

.prompt-cursor { display: inline-block; width: 8px; height: 13px; background: #7c6af7; animation: blink 1s step-end infinite; border-radius: 1px; margin-left: 4px; vertical-align: middle; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.empty-glyph { font-size: 52px; color: #7c6af7 !important; animation: pulse 3s ease-in-out infinite; display: block; text-align: center; margin-bottom: 12px; }
@keyframes pulse { 0%,100% { text-shadow: 0 0 20px rgba(124,106,247,0.3); } 50% { text-shadow: 0 0 60px rgba(124,106,247,0.9); } }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────
for key, val in {
    "messages":     [],
    "indexed":      False,
    "repo_url":     "",
    "file_count":   0,
    "chunk_count":  0,
    "source_files": [],
    "query_count":  0,
    "repo_summary": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Top bar ───────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="cl-logo">⬡ codelens <span>/ agentic rag</span></div>
    <div class="cl-badge">react agent · hybrid retrieval · rrf fusion</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# NOT INDEXED
# ══════════════════════════════════════════════════════════════════════
if not st.session_state.indexed:

    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1.5rem;">
        <div class="empty-glyph">⬡</div>
        <div style="font-size:24px; font-weight:600; color:#e6edf3; letter-spacing:-0.5px; margin-bottom:8px;">
            ask any codebase anything
        </div>
        <div style="font-size:13px; color:#484f58; line-height:1.8;">
            an ai agent explores the repo for you<br>
            multi-hop reasoning · exact file + line citations
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        github_url = st.text_input(
            "github repository url",
            placeholder="https://github.com/tiangolo/fastapi",
            key="main_url"
        )
        if st.button("⟶  index repository", key="main_index_btn"):
            if not github_url.strip():
                st.error("✗ paste a github url first")
            elif not github_url.startswith("https://github.com/"):
                st.error("✗ must be a github.com url")
            else:
                bar = st.progress(0, text="[ 0%] cloning repository...")
                try:
                    repo_path = clone_repo(github_url, "repo_indexed")
                    files     = collect_files(repo_path)
                    bar.progress(25, text=f"[25%] found {len(files)} files — parsing ast...")
                    chunks = chunk_repo(files)
                    bar.progress(50, text=f"[50%] {len(chunks)} chunks — embedding...")
                    embed_and_save(chunks)
                    clear_cache()
                    bar.progress(75, text="[75%] agent exploring repo...")
                    summary_result = auto_summarize_repo()
                    bar.progress(100, text="[100%] ready")

                    st.session_state.indexed      = True
                    st.session_state.repo_url     = github_url
                    st.session_state.file_count   = len(files)
                    st.session_state.chunk_count  = len(chunks)
                    st.session_state.messages     = []
                    st.session_state.source_files = []
                    st.session_state.query_count  = 0
                    st.session_state.repo_summary = summary_result["summary"]
                    st.success("✓ ready — agent is standing by")
                    st.rerun()

                except Exception as e:
                    bar.empty()
                    st.error(f"✗ {str(e)[:200]}")

    st.markdown("""
    <div style="display:flex; gap:10px; justify-content:center; margin-top:2rem; flex-wrap:wrap;">
        <div style="background:#161b22;border:1px solid #21262d;border-radius:20px;padding:6px 16px;font-size:11px;color:#8b949e;">
            <span style="color:#7c6af7;margin-right:6px;font-weight:600;">01</span>paste github url
        </div>
        <div style="background:#161b22;border:1px solid #21262d;border-radius:20px;padding:6px 16px;font-size:11px;color:#8b949e;">
            <span style="color:#7c6af7;margin-right:6px;font-weight:600;">02</span>agent auto-explores
        </div>
        <div style="background:#161b22;border:1px solid #21262d;border-radius:20px;padding:6px 16px;font-size:11px;color:#8b949e;">
            <span style="color:#7c6af7;margin-right:6px;font-weight:600;">03</span>ask anything
        </div>
    </div>
    <div style="text-align:center;margin-top:1.5rem;font-size:12px;color:#484f58;">
        waiting for repo <span class="prompt-cursor"></span>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════
# INDEXED — main interface
# ══════════════════════════════════════════════════════════════════════

repo_short = st.session_state.repo_url.replace("https://github.com/", "")

st.markdown(f"""
<div class="repo-tag">
    <span>⬡</span>
    <div>
        <div class="repo-tag-name">~ {repo_short}</div>
        <div class="repo-tag-meta">agentic rag · multi-hop reasoning · hybrid retrieval</div>
    </div>
</div>
<div class="stat-row">
    <div class="stat-card"><div class="stat-num">{st.session_state.file_count}</div><div class="stat-label">files indexed</div></div>
    <div class="stat-card"><div class="stat-num">{st.session_state.chunk_count}</div><div class="stat-label">code chunks</div></div>
    <div class="stat-card"><div class="stat-num">{st.session_state.query_count}</div><div class="stat-label">queries asked</div></div>
    <div class="stat-card"><div class="stat-num">ReAct</div><div class="stat-label">agent mode</div></div>
</div>
""", unsafe_allow_html=True)

if st.session_state.repo_summary:
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">⬡ agent auto-summary</div>
        {st.session_state.repo_summary}
    </div>
    """, unsafe_allow_html=True)

with st.expander("▸ index a different repository"):
    new_url = st.text_input("new github url", placeholder="https://github.com/user/repo", key="new_url")
    if st.button("⟶  re-index", key="reindex_btn"):
        if new_url.strip() and new_url.startswith("https://github.com/"):
            bar = st.progress(0, text="[ 0%] cloning...")
            try:
                repo_path = clone_repo(new_url, "repo_indexed")
                files     = collect_files(repo_path)
                bar.progress(25, text=f"[25%] {len(files)} files — chunking...")
                chunks = chunk_repo(files)
                bar.progress(50, text="[50%] embedding...")
                embed_and_save(chunks)
                clear_cache()
                bar.progress(75, text="[75%] agent exploring...")
                summary_result = auto_summarize_repo()
                bar.progress(100, text="[100%] done")
                st.session_state.repo_url     = new_url
                st.session_state.file_count   = len(files)
                st.session_state.chunk_count  = len(chunks)
                st.session_state.messages     = []
                st.session_state.source_files = []
                st.session_state.query_count  = 0
                st.session_state.repo_summary = summary_result["summary"]
                st.success("✓ new repo indexed")
                st.rerun()
            except Exception as e:
                st.error(f"✗ {str(e)[:150]}")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ── Helper: render faithfulness bar ──────────────────────────────────
def render_faith_bar(msg: dict):
    faith       = msg.get("faith", 80)
    faith_color = msg.get("faith_color", "#3fb950")
    faith_label = msg.get("faith_label", "Good")
    faith_bad   = msg.get("faith_unsupported", [])

    unsupported_html = ""
    if faith_bad:
        items = " · ".join(faith_bad[:2])
        unsupported_html = f'<div class="faith-unsupported">⚠ unsupported: {items}</div>'

    st.markdown(f"""
    <div class="faith-row">
        <span class="faith-lbl">faithfulness</span>
        <div class="faith-track">
            <div style="height:100%;border-radius:2px;width:{faith}%;
            background:{faith_color};box-shadow:0 0 6px {faith_color}88;"></div>
        </div>
        <span class="faith-pct" style="color:{faith_color};">{faith}%</span>
        <span class="faith-label-badge" style="color:{faith_color};
        background:{faith_color}22;border:1px solid {faith_color}44;">{faith_label}</span>
    </div>
    {unsupported_html}
    """, unsafe_allow_html=True)


# ── Chat layout ───────────────────────────────────────────────────────
col_chat, col_right = st.columns([3, 1])

with col_chat:

    # ── Chat history ──
    for msg in st.session_state.messages:
        role_cls   = "user" if msg["role"] == "user" else "assistant"
        role_label = "you"  if msg["role"] == "user" else "codelens agent"
        bubble_cls = "user" if msg["role"] == "user" else ""

        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-role {role_cls}">{role_label}</div>
            <div class="msg-bubble {bubble_cls}">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # Thinking panel
        if msg.get("steps"):
            tool_steps = [s for s in msg["steps"] if s["type"] == "tool"]
            think_html = '<div class="thinking-panel"><div class="thinking-title">agent reasoning</div>'
            for s in msg["steps"]:
                if s["type"] == "tool":
                    result_preview = ""
                    if s.get("result"):
                        first_line = str(s["result"]).split("\n")[0][:80]
                        result_preview = f'<div class="think-result">→ {first_line}</div>'
                    think_html += f"""
                    <div class="think-step">
                        <div class="think-icon tool">⟳</div>
                        <div class="think-body">
                            <div class="think-tool">{s.get("tool","tool")}()</div>
                            <div class="think-thought">{s.get("thought","")[:100]}</div>
                            {result_preview}
                        </div>
                    </div>"""
                else:
                    think_html += f"""
                    <div class="think-step">
                        <div class="think-icon final">✓</div>
                        <div class="think-body">
                            <div class="think-tool final">final answer</div>
                            <div class="think-thought">{s.get("thought","")[:100]}</div>
                        </div>
                    </div>"""
            think_html += f'<div class="step-badge">{len(tool_steps)} tool calls · {len(msg["steps"])} total steps</div>'
            think_html += '</div>'
            st.markdown(think_html, unsafe_allow_html=True)

        # Source chips + faithfulness
        if msg.get("sources"):
            chips = "".join([
                f'<span class="chip">{s["filepath"].replace(chr(92),"/").split("/")[-1]}:{s["start_line"]}</span>'
                for s in msg["sources"]
            ])
            st.markdown(f'<div class="chips-row">{chips}</div>', unsafe_allow_html=True)

            # ── REAL faithfulness bar ──
            render_faith_bar(msg)

            with st.expander(f"▸ view {len(msg['sources'])} source snippets"):
                for s in msg["sources"]:
                    lang = s["filepath"].split(".")[-1]
                    st.caption(f"{s['filepath']} · lines {s['start_line']}–{s['end_line']}")
                    st.code(
                        s.get("text","")[:900],
                        language=lang if lang in ["python","javascript","typescript","java","go"] else "text"
                    )

    # ── Quick questions ──
    if not st.session_state.messages:
        st.markdown('<div class="quick-label">try asking —</div>', unsafe_allow_html=True)
        quick_qs = [
            "what does this project do?",
            "how does routing work?",
            "where is error handling?",
            "how does auth work?",
            "explain the main entry point",
            "what design patterns are used?",
        ]
        c1, c2, c3 = st.columns(3)
        for i, q in enumerate(quick_qs):
            with [c1, c2, c3][i % 3]:
                if st.button(q, key=f"qq_{i}"):
                    st.session_state._pending_q = q
                    st.rerun()

    # ── Handle quick question ──
    if hasattr(st.session_state, "_pending_q"):
        question = st.session_state._pending_q
        del st.session_state._pending_q
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.query_count += 1

        with st.spinner(f"[ agent ] {question[:60]}..."):
            result = run_agent(question, chat_history=st.session_state.messages[:-1])

        # ── REAL faithfulness scoring ──
        with st.spinner("[ evaluating faithfulness ]..."):
            faith_result = score_faithfulness(question, result["answer"], result["sources"])

        st.session_state.messages.append({
            "role":              "assistant",
            "content":           result["answer"],
            "steps":             result["steps"],
            "sources":           result["sources"],
            "faith":             faith_result["percentage"],
            "faith_label":       faith_result["label"],
            "faith_color":       faith_result["color"],
            "faith_unsupported": faith_result["unsupported"],
        })
        st.session_state.source_files = [s["filepath"] for s in result["sources"]]
        st.rerun()

    # ── Chat input ──
    if question := st.chat_input("▸  ask the agent anything about the codebase..."):
        if not os.path.exists("data/index.faiss"):
            st.error("✗ index not found — re-index the repo")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.query_count += 1

            with st.spinner(f"[ agent thinking ] {question[:60]}..."):
                result = run_agent(question, chat_history=st.session_state.messages[:-1])

            # ── REAL faithfulness scoring ──
            with st.spinner("[ evaluating faithfulness ]..."):
                faith_result = score_faithfulness(question, result["answer"], result["sources"])

            st.session_state.messages.append({
                "role":              "assistant",
                "content":           result["answer"],
                "steps":             result["steps"],
                "sources":           result["sources"],
                "faith":             faith_result["percentage"],
                "faith_label":       faith_result["label"],
                "faith_color":       faith_result["color"],
                "faith_unsupported": faith_result["unsupported"],
            })
            st.session_state.source_files = [s["filepath"] for s in result["sources"]]
            st.rerun()


# ── Right panel ───────────────────────────────────────────────────────
with col_right:

    if st.session_state.source_files:
        st.markdown("""
        <div style="font-size:9px;font-weight:600;letter-spacing:.1em;
        color:#484f58;text-transform:uppercase;margin-bottom:10px;">
        retrieved files
        </div>""", unsafe_allow_html=True)

        seen = {}
        for f in st.session_state.source_files:
            fname = f.replace("\\", "/").split("/")[-1]
            seen[fname] = seen.get(fname, 0) + 1

        for i, (fname, hits) in enumerate(seen.items()):
            border = "#3d3580" if i == 0 else "#21262d"
            color  = "#7c6af7" if i == 0 else "#58a6ff"
            hits_html = f"<div style='font-size:9px;color:#484f58;margin-top:2px;'>× {hits} hits</div>" if hits > 1 else ""
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid {border};border-radius:6px;padding:8px 10px;margin-bottom:5px;">
                <div style="font-size:11px;font-weight:600;color:{color};">{fname}</div>
                {hits_html}
            </div>""", unsafe_allow_html=True)

    last_ai = next(
        (m for m in reversed(st.session_state.messages)
         if m["role"] == "assistant" and m.get("steps")), None
    )

    if last_ai and last_ai.get("steps"):
        st.markdown("""
        <div style="font-size:9px;font-weight:600;letter-spacing:.1em;
        color:#484f58;text-transform:uppercase;margin:14px 0 8px;">
        last agent run
        </div>""", unsafe_allow_html=True)

        tool_calls = [s for s in last_ai["steps"] if s["type"] == "tool"]
        for s in tool_calls:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:6px;padding:7px 10px;margin-bottom:4px;">
                <div style="font-size:10px;font-weight:600;color:#7c6af7;">{s.get("tool","tool")}()</div>
                <div style="font-size:9px;color:#484f58;margin-top:2px;line-height:1.4;">{str(s.get("args",{}))[:60]}</div>
            </div>""", unsafe_allow_html=True)

        # ── Show faithfulness in right panel too ──
        if last_ai.get("faith"):
            faith       = last_ai["faith"]
            faith_color = last_ai.get("faith_color", "#3fb950")
            faith_label = last_ai.get("faith_label", "Good")
            st.markdown(f"""
            <div style="margin-top:12px;background:#161b22;border:1px solid #21262d;
            border-radius:6px;padding:10px 12px;">
                <div style="font-size:9px;color:#484f58;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:6px;">faithfulness</div>
                <div style="font-size:22px;font-weight:600;color:{faith_color};">{faith}%</div>
                <div style="font-size:10px;color:{faith_color};margin-top:2px;">{faith_label}</div>
                <div style="height:3px;background:#21262d;border-radius:2px;margin-top:8px;overflow:hidden;">
                    <div style="height:100%;width:{faith}%;background:{faith_color};
                    box-shadow:0 0 6px {faith_color}88;border-radius:2px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    if not st.session_state.source_files:
        st.markdown("""
        <div style="font-size:10px;color:#484f58;text-align:center;
        margin-top:2rem;line-height:2.2;text-transform:uppercase;letter-spacing:.06em;">
        agent sources<br>appear here<br>after a query
        </div>""", unsafe_allow_html=True)