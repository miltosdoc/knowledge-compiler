"""FROZEN v1 module - kept for reference only.

Superseded by the kc2/ package. Paths here are the original hardcoded
macOS paths and the stages do not connect (sieve emits .txt, distiller
reads .json). Do not build on this; see PLAN_v2.md.
The API key formerly hardcoded here was leaked publicly and must be
treated as compromised; credentials now come from the environment.
"""
import os

"""
Knowledge Compiler - Phase 2: Graph Indexer
Scans vault/atomic/*.md files, extracts [[Links]] and #tags,
builds an adjacency list (graph.json) for the knowledge graph.
"""
import re
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/Users/meditalks/knowledge-compiler")
ATOMIC_DIR = BASE_DIR / "vault" / "atomic"
OUTPUT_FILE = BASE_DIR / "vault" / "graph.json"

def parse_atomic_note(content: str, filename: str):
    """Parse a single atomic note into structured data."""
    notes = []
    # Split by the Zettelkasten delimiter
    blocks = re.split(r'^---\s*$', content.strip(), flags=re.MULTILINE)
    
    current_note = {}
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Extract title
        title_match = re.search(r'^title:\s*(.+)$', block, re.MULTILINE)
        # Extract tags
        tags_match = re.search(r'^tags:\s*(.+)$', block, re.MULTILINE)
        # Extract content (everything after the second --- that isn't title/tags)
        # Extract links
        links = re.findall(r'\[\[(.+?)\]\]', block)
        
        if title_match:
            if current_note and 'title' in current_note:
                notes.append(current_note.copy())
            current_note = {
                'title': title_match.group(1).strip(),
                'tags': [],
                'content': '',
                'links': links,
                'source_file': filename
            }
            if tags_match:
                current_note['tags'] = [t.strip() for t in tags_match.group(1).split(',')]
        elif current_note:
            # Accumulate content
            if not title_match and block and 'title:' not in block:
                current_note['content'] += block + '\n'
                # Re-extract links from accumulated content
                current_note['links'] = list(set(current_note['links'] + re.findall(r'\[\[(.+?)\]\]', block)))
    
    if current_note and 'title' in current_note:
        notes.append(current_note)
    
    # If no Zettelkasten structure found, treat the whole file as one note
    if not notes:
        links = re.findall(r'\[\[(.+?)\]\]', content)
        tags = re.findall(r'#(\w+)', content)
        title_match = re.search(r'^title:\s*(.+)$', content, re.MULTILINE)
        notes.append({
            'title': title_match.group(1).strip() if title_match else filename.replace('.md', ''),
            'tags': tags,
            'content': content,
            'links': links,
            'source_file': filename
        })
    
    return notes

def build_graph():
    """Build the full knowledge graph from all atomic notes."""
    files = list(ATOMIC_DIR.glob("*.md"))
    print(f"Scanning {len(files)} atomic notes...")
    
    all_notes = []
    adjacency = defaultdict(set)  # title -> set of linked titles
    tag_index = defaultdict(list)  # tag -> list of note titles
    title_to_file = {}  # title -> source file
    
    for f in files:
        content = f.read_text(encoding='utf-8')
        notes = parse_atomic_note(content, f.name)
        
        for note in notes:
            all_notes.append(note)
            title = note['title']
            title_to_file[title] = f.name
            
            # Build adjacency (bidirectional)
            for link in note['links']:
                adjacency[title].add(link)
                adjacency[link].add(title)  # bidirectional
            
            # Build tag index
            for tag in note['tags']:
                tag_index[tag].append(title)
    
    # Calculate stats
    total_links = sum(len(v) for v in adjacency.values())
    hub_notes = sorted(adjacency.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    
    graph_data = {
        "metadata": {
            "total_notes": len(all_notes),
            "total_links": total_links,
            "total_tags": len(tag_index),
            "total_files": len(files),
            "top_hubs": [(title, len(links)) for title, links in hub_notes]
        },
        "notes": all_notes,
        "adjacency": {k: list(v) for k, v in adjacency.items()},
        "tag_index": {k: v for k, v in tag_index.items()},
        "title_to_file": title_to_file
    }
    
    # Save graph
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== GRAPH INDEXER RESULTS ===")
    print(f"Notes parsed:      {len(all_notes)}")
    print(f"Unique concepts:   {len(adjacency)}")
    print(f"Total links:       {total_links}")
    print(f"Tags found:        {len(tag_index)}")
    print(f"\nTop intuition hubs:")
    for title, count in hub_notes[:10]:
        print(f"  [{count} links] {title}")
    print(f"\nTag distribution:")
    for tag, titles in sorted(tag_index.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"  {tag}: {len(titles)} notes")
    print(f"\nGraph saved to: {OUTPUT_FILE}")
    
    return graph_data

if __name__ == "__main__":
    build_graph()