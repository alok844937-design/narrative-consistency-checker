"""
Main execution script for Narrative Consistency Checker.
IIT Kharagpur Data Science Hackathon 2026
"""
import pandas as pd
import argparse
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import NarrativeConsistencyPipeline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Narrative Consistency Checker')
    parser.add_argument('--input', type=str, default='data/sample_input.csv',
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='output/results.json',
                       help='Path to output JSON file')
    parser.add_argument('--report', type=str, default='output/report.txt',
                       help='Path to output report file')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Chunk size for text splitting')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of contexts to retrieve per claim')
    parser.add_argument('--detailed', action='store_true',
                       help='Include detailed results in output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NARRATIVE CONSISTENCY CHECKER")
    print("IIT Kharagpur Data Science Hackathon 2026")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load input data
    print(f"\nLoading input from: {args.input}")
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Validate required columns
    required_columns = ['narrative', 'reference']  # Adjust based on actual CSV format
    if not all(col in df.columns for col in required_columns):
        print(f"Warning: Expected columns {required_columns} not found in CSV")
        print(f"Available columns: {df.columns.tolist()}")
        print("\nAssuming first column is narrative and second is reference...")
        if len(df.columns) >= 2:
            narrative_col = df.columns[0]
            reference_col = df.columns[1]
        else:
            print("Error: CSV must have at least 2 columns")
            return
    else:
        narrative_col = 'narrative'
        reference_col = 'reference'
    
    # Initialize pipeline
    print("\n" + "="*60)
    pipeline = NarrativeConsistencyPipeline(
        chunk_size=args.chunk_size,
        top_k_retrieval=args.top_k
    )
    
    # Process each row (or first row for demo)
    print("\n" + "="*60)
    print("PROCESSING DATA")
    print("="*60)
    
    # For hackathon, typically process first row or all rows
    # Modify this based on requirements
    
    all_results = []
    
    for idx, row in df.iterrows():
        print(f"\n{'='*60}")
        print(f"Processing Entry {idx + 1}/{len(df)}")
        print(f"{'='*60}")
        
        narrative_text = str(row[narrative_col])
        reference_text = str(row[reference_col])
        
        # Run consistency check
        results = pipeline.check_consistency(
            narrative_text=narrative_text,
            reference_text=reference_text,
            detailed=args.detailed
        )
        
        results['entry_id'] = idx
        all_results.append(results)
        
        # Generate and save report for this entry
        report = pipeline.generate_report(results)
        print(report)
        
        # If processing multiple entries, you might want to save individual reports
        if len(df) > 1:
            entry_report_path = args.report.replace('.txt', f'_entry_{idx}.txt')
            with open(entry_report_path, 'w', encoding='utf-8') as f:
                f.write(report)
    
    # Save all results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    if len(all_results) == 1:
        # Single result
        pipeline.save_results(all_results[0], args.output)
        
        # Save report
        report = pipeline.generate_report(all_results[0])
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {args.report}")
    else:
        # Multiple results
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"All results saved to: {args.output}")
        
        # Create summary report
        summary_report = f"""
{'='*60}
BATCH PROCESSING SUMMARY
{'='*60}

Total Entries Processed: {len(all_results)}

Average Consistency Score: {sum(r['summary']['consistency_score'] for r in all_results)/len(all_results):.2%}

Individual Entry Scores:
"""
        for i, result in enumerate(all_results):
            summary_report += f"Entry {i+1}: {result['summary']['consistency_percentage']}%\n"
        
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        print(f"Summary report saved to: {args.report}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()