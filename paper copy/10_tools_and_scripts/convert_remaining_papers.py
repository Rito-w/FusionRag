#!/usr/bin/env python3
"""
Convert remaining PDF papers to text format for analysis
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def convert_pdf_to_text(pdf_path, output_path):
    """Convert PDF to text using mineru"""
    try:
        # Create a temporary directory for output
        temp_dir = output_path.parent / "temp_conversion"
        temp_dir.mkdir(exist_ok=True)

        # Use mineru to convert PDF to markdown
        cmd = [
            "magic-pdf",
            "pdf-command",
            "--pdf", str(pdf_path),
            "--method", "auto"
        ]

        # Change to temp directory for output
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        os.chdir(original_cwd)

        if result.returncode == 0:
            # Find the generated markdown file in temp directory
            md_files = list(temp_dir.glob("**/*.md"))
            if md_files:
                # Read the markdown content and save as txt
                with open(md_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # Clean up temp directory
                import shutil
                shutil.rmtree(temp_dir)
                return True

        print(f"Failed to convert {pdf_path}: {result.stderr}")
        # Clean up temp directory on failure
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False

    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return False

def main():
    # Load paper1.json to get all papers
    paper1_path = Path("paper1.json")
    with open(paper1_path, 'r', encoding='utf-8') as f:
        papers = json.load(f)

    # Directory paths
    downloads_dir = Path("all_paper1_downloads")
    text_dir = Path("top_papers_text")
    text_dir.mkdir(exist_ok=True)
    
    # Get already converted papers
    existing_texts = set(f.stem.split('_')[0] for f in text_dir.glob("*.txt"))
    
    converted_count = 0
    failed_count = 0
    
    for i, paper in enumerate(papers):
        # Extract paper ID from link
        paper_id = paper['link'].split('/')[-1]
        
        # Skip if already converted
        if paper_id in existing_texts:
            print(f"Skipping {paper_id} - already converted")
            continue
        
        # Find corresponding PDF file
        pdf_files = list(downloads_dir.glob(f"{paper_id}_*.pdf"))
        if not pdf_files:
            print(f"PDF not found for {paper_id}")
            failed_count += 1
            continue
        
        pdf_path = pdf_files[0]
        
        # Create output text file name
        title_part = paper['title'][:50].replace('/', '_').replace(':', '_')
        output_name = f"{paper_id}_{title_part}.txt"
        output_path = text_dir / output_name
        
        print(f"Converting {i+1}/{len(papers)}: {paper_id}")
        
        if convert_pdf_to_text(pdf_path, output_path):
            converted_count += 1
            print(f"✓ Converted: {output_name}")
        else:
            failed_count += 1
            print(f"✗ Failed: {paper_id}")
    
    print(f"\nConversion complete:")
    print(f"Converted: {converted_count}")
    print(f"Failed: {failed_count}")
    print(f"Already existed: {len(existing_texts)}")

if __name__ == "__main__":
    main()
