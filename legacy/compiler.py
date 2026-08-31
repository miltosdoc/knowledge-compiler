"""FROZEN v1 module - kept for reference only.

Superseded by the kc2/ package. Paths here are the original hardcoded
macOS paths and the stages do not connect (sieve emits .txt, distiller
reads .json). Do not build on this; see PLAN_v2.md.
The API key formerly hardcoded here was leaked publicly and must be
treated as compromised; credentials now come from the environment.
"""
import os

"""
Knowledge Compiler - Phase 3: The Compiler
Takes a "Seed" concept, traverses the knowledge graph,
and compiles a high-density "Holographic Prompt" (~10k tokens).
"""
import json
from pathlib import Path
from collections import deque

BASE_DIR = Path("/Users/meditalks/knowledge-compiler")
GRAPH_FILE = BASE_DIR / "vault" / "graph.json"
ATOMIC_DIR = BASE_DIR / "vault" / "atomic"

# Rough token estimation: ~4 chars per token for English/mixed content
CHARS_PER_TOKEN = 4
MAX_PROMPT_TOKENS = 10000

def load_graph():
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN

def traverse_graph(seed: str, graph: dict, max_depth: int = 2, max_notes: int = 50):
    """
    BFS traversal from a seed concept.
    Returns an ordered list of note titles, prioritized by proximity to seed.
    """
    adjacency = graph.get("adjacency", {})
    notes_by_title = {}
    for note in graph.get("notes", []):
        notes_by_title[note["title"]] = note
    
    visited = set()
    queue = deque([(seed, 0)])
    result = []
    
    # Also add notes matching the seed by tag
    tag_index = graph.get("tag_index", {})
    for tag, titles in tag_index.items():
        if seed.lower() in tag.lower():
            for t in titles:
                if t not in visited:
                    queue.append((t, 0))
    
    # Also add notes matching the seed by title substring
    for title in adjacency.keys():
        if seed.lower() in title.lower() and title not in visited:
            queue.append((title, 0))
    
    while queue and len(result) < max_notes:
        current, depth = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        result.append((current, depth))
        
        if depth < max_depth:
            neighbors = adjacency.get(current, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
    
    return result, notes_by_title

def compile_prompt(seed: str, graph: dict = None, max_tokens: int = MAX_PROMPT_TOKENS):
    """
    Compile a Holographic Prompt from a seed concept.
    Returns a high-density system prompt containing the distilled clinical intuition.
    """
    if graph is None:
        graph = load_graph()
    
    traversed, notes_by_title = traverse_graph(seed, graph)
    
    if not traversed:
        return f"# Clinical Intuition: {seed}\n\nNo distilled intuition found for this concept. The knowledge graph may not contain relevant notes yet."
    
    # Build the prompt, prioritizing closer notes
    prompt_parts = [
        f"# Clinical Intuition Module: {seed}",
        f"# Compiled from {len(traversed)} connected concepts in the knowledge graph.",
        "",
        "The following represents distilled clinical intuition from 1,200+ patient encounters.",
        "This is NOT reference text. This is compiled reasoning weight.",
        "",
        "---",
        ""
    ]
    
    current_tokens = estimate_tokens("\n".join(prompt_parts))
    notes_included = 0
    
    # Group by depth for structured output
    by_depth = {}
    for title, depth in traversed:
        by_depth.setdefault(depth, []).append(title)
    
    for depth in sorted(by_depth.keys()):
        if current_tokens >= max_tokens:
            break
        
        depth_label = "PRIMARY" if depth == 0 else f"ASSOCIATED (depth {depth})"
        section_header = f"\n## {depth_label} INTUITION\n"
        prompt_parts.append(section_header)
        current_tokens += estimate_tokens(section_header)
        
        for title in by_depth[depth]:
            if current_tokens >= max_tokens:
                break
            
            note = notes_by_title.get(title)
            if not note:
                # Try to load from file
                source = graph.get("title_to_file", {}).get(title)
                if source:
                    filepath = ATOMIC_DIR / source
                    if filepath.exists():
                        content = filepath.read_text(encoding='utf-8')
                        note = {"title": title, "content": content, "tags": [], "links": []}
            
            if note:
                tags_str = " ".join(note.get("tags", []))
                links_str = ", ".join(note.get("links", []))
                note_block = f"### {note['title']}\n"
                if tags_str:
                    note_block += f"Tags: {tags_str}\n"
                note_block += f"{note.get('content', '').strip()}\n"
                if links_str:
                    note_block += f"Connected to: {links_str}\n"
                note_block += "\n"
                
                note_tokens = estimate_tokens(note_block)
                if current_tokens + note_tokens > max_tokens:
                    break
                
                prompt_parts.append(note_block)
                current_tokens += note_tokens
                notes_included += 1
    
    # Add footer
    footer = f"\n---\n# END OF COMPILED INTUITION | {notes_included} concepts | ~{current_tokens} tokens"
    prompt_parts.append(footer)
    
    compiled = "\n".join(prompt_parts)
    
    print(f"\n=== COMPILER OUTPUT ===")
    print(f"Seed:              {seed}")
    print(f"Concepts found:    {len(traversed)}")
    print(f"Concepts included: {notes_included}")
    print(f"Estimated tokens:  ~{current_tokens}")
    print(f"Prompt length:     {len(compiled)} chars")
    
    return compiled

if __name__ == "__main__":
    import sys
    seed = sys.argv[1] if len(sys.argv) > 1 else "Heart Failure"
    compiled = compile_prompt(seed)
    
    # Save the compiled prompt
    output_path = BASE_DIR / "vault" / "compiled" / f"{seed.replace(' ', '_')}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(compiled, encoding='utf-8')
    print(f"\nCompiled prompt saved to: {output_path}")
