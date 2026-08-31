"""FROZEN v1 module - kept for reference only.

Superseded by the kc2/ package. Paths here are the original hardcoded
macOS paths and the stages do not connect (sieve emits .txt, distiller
reads .json). Do not build on this; see PLAN_v2.md.
The API key formerly hardcoded here was leaked publicly and must be
treated as compromised; credentials now come from the environment.
"""
import os

import os
import json
import time
from pathlib import Path
from openai import OpenAI

# --- CONFIGURATION ---
BASE_DIR = Path("/Users/meditalks/knowledge-compiler")
INPUT_DIR = BASE_DIR / "vault" / "raw"
OUTPUT_DIR = BASE_DIR / "vault" / "atomic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = BASE_DIR / "distiller_log.txt"

client = OpenAI(
    api_key=os.environ["XSILICO_API_KEY"], 
    base_url="https://staging.xsilico.ai/api/v1"
)
MODEL = "Gemma4-31b"

SOCRATIC_MIRROR_PROMPT = """
You are a Medical Epistemologist & Clinical Intuition Analyst.
Your task is to reconstruct the "Latent Reasoning" (Chain of Thought) of the cardiologist.

INPUTS:
1. Transcript: The raw dialogue between doctor and patient.
2. Notes: The final clinical summary.

ANALYSIS PROTOCOL:
1. Anchor Point Detection: Identify specific phrases in the transcript that triggered a clinical suspicion.
2. Implicit Knowledge Mapping: Identify the medical "priors" the doctor used that are not explicitly stated.
3. The Intuition Path: Map the trajectory: Dialogue Trigger -> Pattern Recognition -> Differential Diagnosis -> Final Conclusion.
4. Synthesis into Atomic Principles: Convert this specific case into a general "Intuition Rule".

OUTPUT FORMAT:
Generate a set of Atomic Notes in Zettelkasten format.
Each note must be a separate Markdown block.
Format for each note:
---
title: [Concise Clinical Principle]
tags: [#protocol, #vocabulary, #intuition]
---
Content: [The distilled reasoning and the 'why']
Links: [[Related Concept]]
---
"""

def distill_transcript(transcript_data):
    if isinstance(transcript_data, dict):
        transcript = transcript_data.get("transcript", "")
        notes = transcript_data.get("notes", "")
    else:
        return None
    
    if not transcript or not notes:
        return None
    
    user_content = f"TRANSCRIPT:\n{transcript}\n\nNOTES:\n{notes}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SOCRATIC_MIRROR_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return None

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

def main():
    files = sorted(list(INPUT_DIR.glob("*.json")))
    total = len(files)
    
    # Skip already-distilled files (resume support)
    already_done = {f.stem for f in OUTPUT_DIR.glob("*.md")}
    remaining = [f for f in files if f.stem not in already_done]
    
    remaining_100 = remaining[:100]
    print(f"Total: {total} | Already done: {len(already_done)} | Processing: {len(remaining_100)}")
    log(f"BATCH START | Total: {total} | Done: {len(already_done)} | Processing: {len(remaining_100)} | Model: {MODEL}")
    
    success = 0
    errors = 0
    skipped = 0
    
    for i, file in enumerate(remaining_100):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"[{len(already_done)+i+1}/{total}] Distilling {file.name}...", flush=True)
            distilled_content = distill_transcript(data)
            
            if distilled_content:
                output_file = OUTPUT_DIR / f"{file.stem}.md"
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    f_out.write(distilled_content)
                success += 1
                if success % 50 == 0:
                    log(f"MILESTONE | {success} distilled, {errors} errors, {skipped} skipped")
            else:
                skipped += 1
                
        except Exception as e:
            errors += 1
            print(f"Error on {file.name}: {e}")
            log(f"ERROR | {file.name} | {e}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    print(f"\nDONE. Success: {success} | Errors: {errors} | Skipped: {skipped}")
    log(f"BATCH COMPLETE | Success: {success} | Errors: {errors} | Skipped: {skipped}")

if __name__ == "__main__":
    main()
