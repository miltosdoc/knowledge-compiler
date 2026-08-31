"""FROZEN v1 module - kept for reference only.

Superseded by the kc2/ package. Paths here are the original hardcoded
macOS paths and the stages do not connect (sieve emits .txt, distiller
reads .json). Do not build on this; see PLAN_v2.md.
The API key formerly hardcoded here was leaked publicly and must be
treated as compromised; credentials now come from the environment.
"""
import os

import re
import os
from pathlib import Path

def extract_transcripts_to_files(sql_file, output_dir, limit=5):
    """
    Extracts raw transcripts and saves them to individual files for the agent to process.
    """
    vault_path = Path(output_dir)
    raw_dir = vault_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    start_line = -1
    for i, line in enumerate(lines):
        if "COPY public.transcriptions" in line:
            start_line = i + 1
            break
    
    if start_line == -1:
        print("Could not find transcriptions table start.")
        return

    count = 0
    for line in lines[start_line:]:
        if line.startswith("COPY") or line.startswith("INSERT") or line.startswith("ALTER") or line.startswith("SET"):
            break
        
        parts = line.split('\t')
        if len(parts) >= 4:
            t_id = parts[0]
            t_text = parts[3]
            
            # Save each raw transcript to a file
            with open(raw_dir / f"{t_id}.txt", 'w', encoding='utf-8') as f_out:
                f_out.write(t_text)
            
            count += 1
            if count >= limit:
                break
                
    print(f"Extracted {count} raw transcripts to {raw_dir}")

if __name__ == "__main__":
    extract_transcripts_to_files(
        sql_file="database-fresh-dump-2026-04-11T07-49-14Z.sql", 
        output_dir="vault"
    )
