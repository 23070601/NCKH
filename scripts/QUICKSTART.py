#!/usr/bin/env python3
"""
Quick Start Guide - Run Data Collection Pipeline
Predicting Volatility and Risk Level of FDI Stock Prices
"""

import os
import sys
import subprocess

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def main():
    print_section("FDI STOCK DATA COLLECTION - QUICK START")
    
    # Step 1: Verify setup
    print("Step 1: Verifying Project Setup...")
    print("-" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        'data/fdi_stocks_list.csv',
        'src/VNStocks.py',
        'src/utils.py',
        'notebooks/0_data_collection.ipynb',
        'notebooks/1_data_preparation.ipynb'
    ]
    
    all_exist = True
    for file in required_files:
        filepath = os.path.join(base_dir, file)
        status = "✅" if os.path.exists(filepath) else "❌"
        print(f"  {status} {file}")
        if not os.path.exists(filepath):
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some required files are missing!")
        sys.exit(1)
    
    print("\n✅ All required files found!\n")
    
    # Step 2: Check stock list
    print("Step 2: Checking FDI Stock List...")
    print("-" * 70)
    
    stocks_file = os.path.join(base_dir, 'data/fdi_stocks_list.csv')
    with open(stocks_file) as f:
        num_stocks = len(f.readlines()) - 1  # Subtract header
    
    print(f"  ✅ Found {num_stocks} FDI stocks in list")
    print(f"  📊 Sectors covered: Consumer, Finance, Energy, Materials, IT, Healthcare, RE")
    print(f"  🏦 Exchanges: HOSE (primary), HNX (secondary)")
    print(f"  🌐 FDI Status: High/Medium/Low involvement")
    
    # Step 3: Dependency check
    print("\n\nStep 3: Checking Python Dependencies...")
    print("-" * 70)
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'matplotlib': 'Visualization (optional)',
        'sklearn': 'Machine learning utilities',
        'torch': 'PyTorch (for GNN - install later)'
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package:15} - {description}")
        except ImportError:
            print(f"  ❌ {package:15} - {description}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"\n  Install with: pip install -r requirements.txt")
        print(f"  Or: pip install pandas numpy matplotlib scikit-learn\n")
    
    # Step 4: Display next steps
    print("\n\nStep 4: Next Steps")
    print("-" * 70)
    
    print("""
✅ Your project structure is ready!

NEXT ACTIONS:

1. RUN DATA COLLECTION (5-10 minutes)
   - Open: notebooks/0_data_collection.ipynb
   - Execute all cells from top to bottom
   - This will generate:
     • data/stocks.csv (100 stocks metadata)
     • data/fundamentals.csv (financial metrics)
     • data/values.csv (daily prices)
     • data/adj.npy (correlation graph)

2. FEATURE ENGINEERING (10-15 minutes)
   - Open: notebooks/1_data_preparation.ipynb
   - Calculate volatility metrics (your target variable!)
   - Engineer features for GNN
   - Create train/test splits

3. BUILD GNN MODELS (2-4 hours)
   - Install: pip install torch torch-geometric
   - Design Temporal Graph Convolutional Network
   - Train on volatility prediction task
   - Evaluate FDI impact on predictions

4. ANALYSIS & VISUALIZATION (2-3 hours)
   - Analyze how FDI status affects volatility
   - Compare sectors
   - Generate publication-quality figures
   - Document findings for thesis

IMPORTANT NOTES:
─────────────────

• Data Collection: Currently uses SAMPLE data for testing
  To use REAL Vietnamese stock data:
  - Install: pip install vnstock
  - Update: change source='manual' to source='vnstock'
  
• Your unique contribution: FDI focus
  - Analyze if FDI involvement reduces volatility
  - Compare GNN predictions across FDI levels
  - Generate emerging market + GNN research insights
  
• Reference project: S&P 100 Analysis with GNNs
  - GitHub: https://github.com/timothewt/SP100AnalysisWithGNNs
  - You're adapting this excellent framework for Vietnamese FDI context

TROUBLESHOOTING:
────────────────

Q: vnstock not available?
A: Use sample data first to test. Real data optional after framework works.

Q: Import errors when running notebook?
A: Run: pip install -r requirements.txt

Q: How many stocks exactly?
A: 100 FDI companies (fdi_stocks_list.csv, 101 lines = 100 + header)

Q: Need more details?
A: Read: COMPLETION_GUIDE.md (detailed implementation roadmap)
""")
    
    print("\n" + "="*70)
    print("  Ready to start? Open notebooks/0_data_collection.ipynb!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
