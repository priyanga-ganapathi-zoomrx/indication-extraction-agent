#!/usr/bin/env python3
"""
Script to clean the firms column in the input CSV file.

Handles:
- Removing <0> (or <digit>) prefix from firms values
- Preserving ;; as the separator for multiple firms

Usage:
    python clean_firms.py [input_path] [output_path]
    
    If no arguments provided, uses default paths.
"""

import csv
import re
import sys
from pathlib import Path


def clean_firms_value(value: str) -> str:
    """Clean a single firms column value.
    
    Args:
        value: Raw firms value (e.g., "<0>Astellas Pharma;;Pfizer")
        
    Returns:
        Cleaned value (e.g., "Astellas Pharma;;Pfizer")
    """
    if not value or not value.strip():
        return ""
    
    # Remove <digit> prefix (e.g., <0>, <1>, <12>)
    cleaned = re.sub(r'^<\d+>', '', value.strip())
    
    return cleaned.strip()


def clean_csv(input_path: str, output_path: str) -> dict:
    """Clean the firms column in the CSV file.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_rows": 0,
        "rows_with_firms": 0,
        "rows_cleaned": 0,
        "multi_firm_rows": 0,
    }
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = []
        
        for row in reader:
            stats["total_rows"] += 1
            
            if 'firms' in row and row['firms']:
                stats["rows_with_firms"] += 1
                original = row['firms']
                cleaned = clean_firms_value(original)
                
                if original != cleaned:
                    stats["rows_cleaned"] += 1
                
                if ';;' in cleaned:
                    stats["multi_firm_rows"] += 1
                
                row['firms'] = cleaned
            
            rows.append(row)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return stats


def main():
    # Default paths
    default_input = "data/drug/input/abstract_titles(in).csv"
    default_output = "data/drug/input/abstract_titles_cleaned.csv"
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # Validate input exists
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Cleaning CSV...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print()
    
    stats = clean_csv(input_path, output_path)
    
    print("Results:")
    print(f"  Total rows:        {stats['total_rows']}")
    print(f"  Rows with firms:   {stats['rows_with_firms']}")
    print(f"  Rows cleaned:      {stats['rows_cleaned']}")
    print(f"  Multi-firm rows:   {stats['multi_firm_rows']}")
    print()
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
