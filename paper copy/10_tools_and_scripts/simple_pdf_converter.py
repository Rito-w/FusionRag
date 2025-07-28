#!/usr/bin/env python3
"""
Simple PDF to text converter using PyPDF2
"""

import json
import os
import sys
from pathlib import Path

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not installed. Installing...")
    os.system("pip install PyPDF2")
    import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

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
    
    # Process papers starting from the 11th (index 10) since we already have the top 10
    for i, paper in enumerate(papers[10:], start=11):  # Start from 11th paper
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
        title_part = paper['title'][:50].replace('/', '_').replace(':', '_').replace('?', '').replace('\\', '_')
        output_name = f"{paper_id}_{title_part}.txt"
        output_path = text_dir / output_name
        
        print(f"Converting {i}/{len(papers)}: {paper_id}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if text and len(text) > 100:  # Basic validation
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                converted_count += 1
                print(f"✓ Converted: {output_name}")
            except Exception as e:
                print(f"✗ Failed to save {paper_id}: {e}")
                failed_count += 1
        else:
            failed_count += 1
            print(f"✗ Failed to extract text: {paper_id}")
    
    print(f"\nConversion complete:")
    print(f"Converted: {converted_count}")
    print(f"Failed: {failed_count}")
    print(f"Already existed: {len(existing_texts)}")

if __name__ == "__main__":
    main()
