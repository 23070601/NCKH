#!/usr/bin/env python3
"""
Hybrid Volatility & Risk Prediction CLI
Real-time predictions for individual stocks or portfolios
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import pickle

# Import inference pipeline directly to avoid full utils package loading
import importlib.util
inference_path = os.path.join(os.path.dirname(__file__), 'src', 'utils', 'inference.py')
spec = importlib.util.spec_from_file_location("inference", inference_path)
inference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_module)
InferencePipeline = inference_module.InferencePipeline


class PredictionCLI:
    """Command-line interface for predictions."""
    
    def __init__(self):
        self.pipeline = InferencePipeline()
        self.pipeline.load_models()
    
    def predict_stock(self, symbol: str):
        """Predict for single stock."""
        print(f"\n🔍 Predicting volatility & risk for {symbol}...")
        
        prediction = self.pipeline.predict_single_stock(symbol)
        
        if 'error' in prediction:
            print(f"❌ Error: {prediction['error']}")
            return
        
        # Display results
        print(self.pipeline.generate_report(prediction))
        
        # Save
        save_path = self.pipeline.save_predictions(prediction)
        print(f"✓ Predictions saved to {save_path}")
    
    def predict_portfolio(self, symbols_file: str = None):
        """Predict for portfolio."""
        print("\n📊 Predicting portfolio volatility & risk...")
        
        symbols = None
        weights = None
        
        if symbols_file:
            # Load portfolio config
            with open(symbols_file, 'r') as f:
                config = json.load(f)
                symbols = config.get('symbols')
                weights = config.get('weights')
        
        prediction = self.pipeline.predict_portfolio(symbols, weights)
        
        if 'error' in prediction:
            print(f"❌ Error: {prediction['error']}")
            return
        
        # Display results
        print(self.pipeline.generate_report(prediction))
        
        # Save
        save_path = self.pipeline.save_predictions(prediction)
        print(f"✓ Predictions saved to {save_path}")
        
        # Statistics
        print("\n📈 PREDICTION STATISTICS")
        print("-" * 50)
        print(f"Stocks analyzed: {prediction['prediction_count']}")
        print(f"Portfolio volatility: {prediction['portfolio_level']['predicted_volatility_5d']:.6f}")
    
    def generate_dashboard(self, output_file: str = 'volatility_dashboard.html'):
        """Generate HTML dashboard."""
        print("\n📊 Generating dashboard...")
        
        # Load latest predictions
        data = self.pipeline.load_latest_data()
        
        if data is None:
            print("❌ No prediction data available")
            return
        
        # Simple HTML dashboard
        html_content = self._create_dashboard_html(data)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"✓ Dashboard saved to {output_file}")
    
    def _create_dashboard_html(self, data: pd.DataFrame) -> str:
        """Create simple HTML dashboard."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Volatility & Risk Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; font-size: 12px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f9f9f9; font-weight: bold; }}
        .risk-low {{ color: green; }}
        .risk-medium {{ color: orange; }}
        .risk-high {{ color: red; }}
        .footer {{ text-align: center; color: #666; margin-top: 30px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Volatility & Risk Prediction Dashboard</h1>
            <p>Vietnamese FDI Stocks Forecast</p>
        </div>
        
        <div class="card">
            <h2>Summary Statistics</h2>
            <div class="metric">
                <div class="metric-value">{len(data)}</div>
                <div class="metric-label">Total Predictions</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['Predicted_Vol_RF'].mean():.6f}</div>
                <div class="metric-label">Avg Volatility</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['Predicted_Risk'].value_counts().get(2, 0)}</div>
                <div class="metric-label">High Risk Stocks</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Top 10 Highest Volatility Predictions</h2>
            <table>
                <tr>
                    <th>Stock</th>
                    <th>Predicted Vol</th>
                    <th>Risk Level</th>
                </tr>
        """
        
        # Add top volatility stocks
        top_vol = data.nlargest(10, 'Predicted_Vol_RF')
        for _, row in top_vol.iterrows():
            risk_name = {0: 'Low', 1: 'Medium', 2: 'High'}.get(row['Predicted_Risk'], 'Unknown')
            risk_class = f'risk-{risk_name.lower()}'
            html_content += f"""
                <tr>
                    <td>{row['Symbol']}</td>
                    <td>{row['Predicted_Vol_RF']:.6f}</td>
                    <td class="{risk_class}">{risk_name}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        
        <div class="footer">
            <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            <p>Data Source: Vietnamese FDI Stocks | Prediction Method: Hybrid Regression + Classification</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Volatility & Risk Prediction System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Predict stock
    predict_parser = subparsers.add_parser('predict', help='Predict for a single stock')
    predict_parser.add_argument('symbol', help='Stock symbol (e.g., VNM, FPT)')
    
    # Predict portfolio
    portfolio_parser = subparsers.add_parser('portfolio', help='Predict for portfolio')
    portfolio_parser.add_argument('--config', help='Portfolio config file (JSON)')
    
    # Generate dashboard
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate dashboard')
    dashboard_parser.add_argument('--output', default='volatility_dashboard.html', help='Output HTML file')
    
    # List stocks
    subparsers.add_parser('list', help='List available stocks')
    
    # Batch predictions
    batch_parser = subparsers.add_parser('batch', help='Batch predictions for all stocks')
    batch_parser.add_argument('--output-dir', default='data/analysis', help='Output directory')
    
    args = parser.parse_args()
    
    cli = PredictionCLI()
    
    if args.command == 'predict':
        cli.predict_stock(args.symbol.upper())
    
    elif args.command == 'portfolio':
        cli.predict_portfolio(args.config)
    
    elif args.command == 'dashboard':
        cli.generate_dashboard(args.output)
    
    elif args.command == 'list':
        data = cli.pipeline.load_latest_data()
        if data is not None:
            print("\nAvailable stocks:")
            for i, symbol in enumerate(data['Symbol'].unique(), 1):
                print(f"  {i:2d}. {symbol}")
        else:
            print("No data available")
    
    elif args.command == 'batch':
        print("\n🔄 Running batch predictions...")
        data = cli.pipeline.load_latest_data()
        
        if data is not None:
            symbols = data['Symbol'].unique()
            predictions = cli.pipeline.predict_portfolio(list(symbols))
            
            save_path = cli.pipeline.save_predictions(predictions, args.output_dir)
            print(f"✓ Batch predictions saved to {save_path}")
            print(f"✓ Analyzed {predictions['prediction_count']} stocks")
        else:
            print("No data available")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
