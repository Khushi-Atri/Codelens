from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import pathlib
import json

# Load language parsers
PY_LANG  = Language(tspython.language(), "python")
JS_LANG  = Language(tsjavascript.language(), "javascript")

# Map file extensions to parsers
def make_parser(lang):
    parser = Parser()
    parser.set_language(lang)
    return parser

PARSERS = {
    '.py':  make_parser(PY_LANG),
    '.js':  make_parser(JS_LANG),
    '.jsx': make_parser(JS_LANG),
}

# Node types that mean "meaningful code unit"
CHUNK_NODE_TYPES = {
    'function_definition',      # Python function
    'class_definition',         # Python class
    'function_declaration',     # JavaScript function
    'method_definition',        # JavaScript class method
    'arrow_function',           # JS arrow function
    'class_declaration',        # JavaScript class
}

def get_function_name(node, source: str) -> str:
    """Try to extract the name of a function or class node."""
    for child in node.children:
        if child.type == 'identifier':
            return source[child.start_byte:child.end_byte]
    return "unknown"

def chunk_text_file(filepath: pathlib.Path, source: str) -> list[dict]:
    """Sliding window chunker for non-code files like .md, .ts, .ipynb."""
    lines = source.splitlines()
    chunks = []
    for i in range(0, len(lines), 40):
        window = lines[i:i+50]
        if any(l.strip() for l in window):
            chunks.append({
                'text': '\n'.join(window),
                'name': f"lines_{i+1}",
                'type': 'text_window',
                'filepath': str(filepath),
                'start_line': i + 1,
                'end_line': min(i + 50, len(lines)),
                'metadata': f"File: {filepath} | Lines: {i+1}-{min(i+50, len(lines))}"
            })
    return chunks

def chunk_notebook(filepath: pathlib.Path) -> list[dict]:
    """Extract code and markdown cells from Jupyter notebooks."""
    chunks = []
    try:
        content = json.loads(filepath.read_text(errors='ignore'))
        cells = content.get('cells', [])
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type', '')
            source = ''.join(cell.get('source', []))
            if not source.strip():
                continue
            chunks.append({
                'text': source,
                'name': f"cell_{i+1}_{cell_type}",
                'type': f"notebook_{cell_type}",
                'filepath': str(filepath),
                'start_line': i + 1,
                'end_line': i + 1,
                'metadata': f"File: {filepath} | Cell {i+1} ({cell_type})"
            })
    except Exception as e:
        print(f"Could not parse notebook {filepath}: {e}")
    return chunks

def chunk_file(filepath: pathlib.Path) -> list[dict]:
    """Parse one file and return list of chunk dicts."""
    source = filepath.read_text(errors='ignore')
    ext = filepath.suffix
    chunks = []

    # Handle Jupyter notebooks specially
    if ext == '.ipynb':
        return chunk_notebook(filepath)

    # Use tree-sitter for Python and JavaScript
    if ext in PARSERS:
        parser = PARSERS[ext]
        tree = parser.parse(bytes(source, 'utf8'))

        def walk(node):
            if node.type in CHUNK_NODE_TYPES:
                chunk_text = source[node.start_byte:node.end_byte]
                name = get_function_name(node, source)

                if chunk_text.count('\n') >= 2:
                    chunks.append({
                        'text': chunk_text,
                        'name': name,
                        'type': node.type,
                        'filepath': str(filepath),
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'metadata': f"File: {filepath} | Name: {name} | Lines: {node.start_point[0]+1}-{node.end_point[0]+1}"
                    })
                for child in node.children:
                    walk(child)
            else:
                for child in node.children:
                    walk(child)

        walk(tree.root_node)

    # Fallback: sliding window for .md, .ts, .txt and anything else
    if not chunks:
        chunks = chunk_text_file(filepath, source)

    return chunks

def chunk_repo(files: list) -> list[dict]:
    """Chunk all files in the repo."""
    all_chunks = []
    for f in files:
        try:
            file_chunks = chunk_file(f)
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks