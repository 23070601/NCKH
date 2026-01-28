#!/usr/bin/env python3
"""
Simple Hybrid Volatility & Risk Prediction CLI
Quick predictions using existing trained models and data
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


class SimplePredictionCLI:
    """Simple command-line interface for volatility predictions."""
    
    def __init__(self):
        """Initialize prediction CLI."""
        self.data_dir = Path('data')
        self.analysis_dir = self.data_dir / 'analysis'
        self.predictions_df = None
        self.stocks_list = None
        self.load_latest_predictions()
    
    def load_latest_predictions(self):
        """Load latest predictions CSV."""
        pred_files = sorted(self.analysis_dir.glob('predictions_*.csv'))
        if pred_files:
            self.predictions_df = pd.read_csv(pred_files[-1])
            self.stocks_list = self.predictions_df['Symbol'].unique()
            print(f"✓ Loaded predictions for {len(self.stocks_list)} stocks")
        else:
            print("⚠ No predictions file found")
    
    def predict(self, symbol: str):
        """Predict for a single stock."""
        if self.predictions_df is None:
            print("❌ No predictions available")
            return
        
        # Get latest row for this stock
        stock_data = self.predictions_df[self.predictions_df['Symbol'] == symbol]
        
        if len(stock_data) == 0:
            print(f"❌ Stock {symbol} not found. Use 'list' to see available stocks.")
            return
        
        # Use the latest record
        row = stock_data.iloc[-1]
        
        # Extract key predictions
        volatility = row['Predicted_Vol_RF']
        volatility_xgb = row['Predicted_Vol_XGB']
        risk_class = int(row['Predicted_Risk'])
        
        # Risk names
        risk_names = {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}
        risk_name = risk_names.get(risk_class, "Unknown")
        
        # Calculate confidence
        vol_agreement = 1.0 - abs(volatility - volatility_xgb) / max(volatility, volatility_xgb, 0.001)
        confidence = max(0.5, min(0.95, vol_agreement))
        
        # Display report
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║          HYBRID VOLATILITY & RISK PREDICTION REPORT                ║
╚════════════════════════════════════════════════════════════════════╝

📍 Stock Information:
   Symbol:           {symbol}
   Date:             {row['Date']}
   Close Price:      ₫{row['Close']:.2f}
   Daily Return:     {row['DailyLogReturn']*100:.2f}%

📊 Volatility Predictions:
   5-Day Volatility: {volatility:.6f} ({volatility*100:.2f}%)
   RF Model:         {volatility:.6f}
   XGB Model:        {volatility_xgb:.6f}
   Model Agreement:  {vol_agreement:.1%}

⚠️  Risk Assessment:
   Risk Level:       {risk_name}
   Confidence:       {confidence:.1%}

📈 Technical Indicators:
   RSI (14):         {row['RSI']:.1f}
   MACD:             {row['MACD']:.4f}
   Vol Lag (1d):     {row['Vol_Lag_1']:.6f}
   Vol MA (20d):     {row['Vol_MA_20']:.6f}

💰 Fundamental Metrics:
   P/E Ratio:        {row['pe_ratio']:.2f}
   P/B Ratio:        {row['pb_ratio']:.2f}
   ROE:              {row['roe']*100:.1f}%
   ROA:              {row['roa']*100:.1f}%
   Beta:             {row['beta']:.2f}

🌍 Macroeconomic Indicators:
   VN-Index:         {row['VN_Index']:.0f}
   USD/VND:          {row['USD_VND']:.2f}
   Policy Rate:      {row['Policy_Rate']:.1f}%
   CPI Inflation:    {row['Inflation_YoY']*100:.1f}%

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
╚════════════════════════════════════════════════════════════════════╝
        """)
        
        # Save prediction
        output_file = self.analysis_dir / f'predictions_hybrid_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            'symbol': symbol,
            'date': str(row['Date']),
            'volatility': {
                'predicted_rf': float(volatility),
                'predicted_xgb': float(volatility_xgb),
                'model_agreement': float(vol_agreement)
            },
            'risk': {
                'class': int(risk_class),
                'name': risk_name.split()[-2:],  # Extract risk level
                'confidence': float(confidence)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Predictions saved to {output_file.name}\n")
    
    def list_stocks(self):
        """List available stocks."""
        if self.stocks_list is None:
            print("❌ No stocks available")
            return
        
        # Group by risk level
        low_risk = []
        medium_risk = []
        high_risk = []
        
        for symbol in sorted(self.stocks_list):
            stock_data = self.predictions_df[self.predictions_df['Symbol'] == symbol].iloc[-1]
            risk_class = int(stock_data['Predicted_Risk'])
            
            if risk_class == 0:
                low_risk.append(symbol)
            elif risk_class == 1:
                medium_risk.append(symbol)
            else:
                high_risk.append(symbol)
        
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║                   AVAILABLE STOCKS FOR PREDICTION                  ║
╚════════════════════════════════════════════════════════════════════╝

🟢 Low Risk Stocks ({len(low_risk)}):
   {', '.join(low_risk[:20])}
   {f'and {len(low_risk)-20} more...' if len(low_risk) > 20 else ''}

🟡 Medium Risk Stocks ({len(medium_risk)}):
   {', '.join(medium_risk[:20])}
   {f'and {len(medium_risk)-20} more...' if len(medium_risk) > 20 else ''}

🔴 High Risk Stocks ({len(high_risk)}):
   {', '.join(high_risk[:20])}
   {f'and {len(high_risk)-20} more...' if len(high_risk) > 20 else ''}

Total: {len(self.stocks_list)} stocks
        """)
    
    def batch_predict(self, output_dir: str = None):
        """Batch predictions for all stocks."""
        if self.predictions_df is None:
            print("❌ No predictions available")
            return
        
        output_dir = Path(output_dir) if output_dir else self.analysis_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 Generating batch predictions for {len(self.stocks_list)} stocks...")
        
        results = {
            'generated': datetime.now().isoformat(),
            'total_stocks': len(self.stocks_list),
            'stocks': {}
        }
        
        risk_stats = {0: 0, 1: 0, 2: 0}
        vol_values = []
        
        for symbol in self.stocks_list:
            stock_data = self.predictions_df[self.predictions_df['Symbol'] == symbol].iloc[-1]
            vol = stock_data['Predicted_Vol_RF']
            risk = int(stock_data['Predicted_Risk'])
            
            results['stocks'][symbol] = {
                'volatility': float(vol),
                'risk_class': risk,
                'risk_name': {0: 'Low', 1: 'Medium', 2: 'High'}[risk]
            }
            
            risk_stats[risk] += 1
            vol_values.append(vol)
        
        # Statistics
        results['statistics'] = {
            'risk_distribution': {
                'low': risk_stats[0],
                'medium': risk_stats[1],
                'high': risk_stats[2]
            },
            'volatility_stats': {
                'mean': float(np.mean(vol_values)),
                'std': float(np.std(vol_values)),
                'min': float(np.min(vol_values)),
                'max': float(np.max(vol_values))
            }
        }
        
        # Save
        output_file = output_dir / f'batch_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"""
✓ Batch predictions complete:

📊 Risk Distribution:
   🟢 Low Risk:      {risk_stats[0]} stocks
   🟡 Medium Risk:   {risk_stats[1]} stocks
   🔴 High Risk:     {risk_stats[2]} stocks

📈 Volatility Statistics:
   Mean:             {np.mean(vol_values):.6f}
   Std Dev:          {np.std(vol_values):.6f}
   Min:              {np.min(vol_values):.6f}
   Max:              {np.max(vol_values):.6f}

✓ Saved to {output_file.name}
        """)
    
    def dashboard(self, output_file: str = None):
        """Generate HTML dashboard."""
        if self.predictions_df is None:
            print("❌ No predictions available")
            return
        
        output_file = Path(output_file) if output_file else Path('volatility_dashboard.html')
        
        print(f"\n🎨 Generating dashboard...")
        
        # Get latest predictions
        latest = self.predictions_df.groupby('Symbol').tail(1)
        
        # Calculate statistics
        low_risk = len(latest[latest['Predicted_Risk'] == 0])
        medium_risk = len(latest[latest['Predicted_Risk'] == 1])
        high_risk = len(latest[latest['Predicted_Risk'] == 2])
        
        mean_vol = latest['Predicted_Vol_RF'].mean()
        std_vol = latest['Predicted_Vol_RF'].std()
        
        # Top high-volatility stocks
        top_high = latest.nlargest(10, 'Predicted_Vol_RF')[['Symbol', 'Predicted_Vol_RF', 'Predicted_Risk']]
        top_low = latest.nsmallest(10, 'Predicted_Vol_RF')[['Symbol', 'Predicted_Vol_RF', 'Predicted_Risk']]
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hybrid Volatility & Risk Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-box .number {{
            font-size: 32px;
            font-weight: bold;
        }}
        .table-container {{
            margin-top: 30px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }}
        th {{
            background: #f5f5f5;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #ddd;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .risk-low {{ color: #28a745; font-weight: bold; }}
        .risk-medium {{ color: #ffc107; font-weight: bold; }}
        .risk-high {{ color: #dc3545; font-weight: bold; }}
        .footer {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Hybrid Volatility & Risk Prediction Dashboard</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>🟢 Low Risk</h3>
                <div class="number">{low_risk}</div>
            </div>
            <div class="stat-box">
                <h3>🟡 Medium Risk</h3>
                <div class="number">{medium_risk}</div>
            </div>
            <div class="stat-box">
                <h3>🔴 High Risk</h3>
                <div class="number">{high_risk}</div>
            </div>
            <div class="stat-box">
                <h3>📊 Mean Volatility</h3>
                <div class="number">{mean_vol:.4f}</div>
            </div>
        </div>
        
        <div class="table-container">
            <h2>📈 Top 10 Highest Volatility Stocks</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Predicted Volatility</th>
                    <th>Risk Level</th>
                </tr>
                {''.join([f"<tr><td>{row['Symbol']}</td><td>{row['Predicted_Vol_RF']:.6f}</td><td class='risk-{['low', 'medium', 'high'][int(row['Predicted_Risk'])]}'>{['Low', 'Medium', 'High'][int(row['Predicted_Risk'])]}</td></tr>" for _, row in top_high.iterrows()])}
            </table>
        </div>
        
        <div class="table-container">
            <h2>📉 Top 10 Lowest Volatility Stocks</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Predicted Volatility</th>
                    <th>Risk Level</th>
                </tr>
                {''.join([f"<tr><td>{row['Symbol']}</td><td>{row['Predicted_Vol_RF']:.6f}</td><td class='risk-{['low', 'medium', 'high'][int(row['Predicted_Risk'])]}'>{['Low', 'Medium', 'High'][int(row['Predicted_Risk'])]}</td></tr>" for _, row in top_low.iterrows()])}
            </table>
        </div>
        
        <div class="footer">
            <p>Hybrid Volatility & Risk Prediction System v1.0 | Data sources: VN-Index, Company Fundamentals, Macroeconomic Indicators</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Dashboard saved to {output_file}")
        print(f"✓ Open in browser: {output_file.absolute()}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Hybrid Volatility & Risk Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Predict single stock
  python predict_simple.py predict VNM
  
  # List all stocks
  python predict_simple.py list
  
  # Generate batch predictions
  python predict_simple.py batch
  
  # Generate dashboard
  python predict_simple.py dashboard --output my_dashboard.html
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict for a single stock')
    predict_parser.add_argument('symbol', help='Stock symbol (e.g., VNM, FPT, VCB)')
    
    # List command
    subparsers.add_parser('list', help='List available stocks')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch predictions for all stocks')
    batch_parser.add_argument('--output-dir', default='data/analysis', help='Output directory')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate HTML dashboard')
    dashboard_parser.add_argument('--output', default='volatility_dashboard.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = SimplePredictionCLI()
    
    # Execute command
    if args.command == 'predict':
        cli.predict(args.symbol.upper())
    elif args.command == 'list':
        cli.list_stocks()
    elif args.command == 'batch':
        cli.batch_predict(args.output_dir)
    elif args.command == 'dashboard':
        cli.dashboard(args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
