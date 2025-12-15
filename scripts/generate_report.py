#!/usr/bin/env python3
"""
Standalone script to generate benchmark reports from existing results.
Use this to regenerate reports without re-running HPO or training.
"""

import argparse
from benchmark_report import generate_benchmark_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports from existing HPO and training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report from default directories
  python generate_report.py
  
  # Specify custom directories
  python generate_report.py --runs-dir runs --output-dir reports
  
  # Generate report from specific runs directory
  python generate_report.py --runs-dir runs
        """
    )
    parser.add_argument("--runs-dir", type=str, default="runs",
                       help="Directory containing HPO results (default: runs)")
    parser.add_argument("--output-dir", type=str, default="reports",
                       help="Output directory for reports (default: reports)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("📊 BENCHMARK REPORT GENERATOR")
    print(f"{'='*80}")
    print(f"\nReading from: {args.runs_dir}")
    print(f"Output to: {args.output_dir}")
    print(f"\nThis will generate reports from existing results.")
    print(f"No training or HPO will be run.\n")
    
    try:
        report_dir = generate_benchmark_report(
            basic_results=None,  # Only load HPO results from JSON files
            runs_dir=args.runs_dir,
            output_dir=args.output_dir
        )
        print(f"\n✅ Report generated successfully!")
        print(f"📁 Location: {report_dir}")
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

