"""
Real-time Inference Pipeline for Hybrid Volatility & Risk Prediction
Handles data loading, preprocessing, model inference, and result formatting
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, Tuple, Optional
import pickle


class InferencePipeline:
    """
    Production-ready inference pipeline for volatility and risk predictions.
    """
    
    def __init__(self, model_dir: str = 'models/trained', 
                 data_dir: str = 'data',
                 config_file: str = 'config/inference_config.json'):
        """
        Initialize inference pipeline.
        
        Args:
            model_dir: Directory containing trained models
            data_dir: Directory containing data files
            config_file: Configuration file path
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.config_file = Path(config_file)
        
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.config = {}
        
        self.load_config()
    
    def load_config(self):
        """Load inference configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config
            self.config = {
                'prediction_horizon': 5,  # 5-day ahead
                'min_confidence': 0.5,
                'feature_set': 'all',
                'ensemble_method': 'average'
            }
    
    def load_models(self, model_names: list = None):
        """
        Load trained models from disk.
        
        Args:
            model_names: List of model names to load
        """
        if model_names is None:
            model_names = ['rf_regressor', 'rf_classifier']
        
        for model_name in model_names:
            # Try direct path first
            model_path = self.model_dir / f'{model_name}.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"✓ Loaded {model_name}")
            else:
                # Look for latest timestamped version
                pattern = f'{model_name}_*.pkl'
                matching_files = sorted(self.model_dir.glob(pattern))
                if matching_files:
                    latest_path = matching_files[-1]  # Get most recent
                    with open(latest_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✓ Loaded {model_name} from {latest_path.name}")
                else:
                    print(f"⚠ Model not found: {model_path}")
    
    def load_latest_data(self) -> pd.DataFrame:
        """
        Load latest prediction data.
        
        Returns:
            DataFrame with features and recent predictions
        """
        # Look for latest predictions file
        analysis_dir = self.data_dir / 'analysis'
        if analysis_dir.exists():
            prediction_files = sorted(analysis_dir.glob('predictions_*.csv'))
            if prediction_files:
                latest_file = prediction_files[-1]
                return pd.read_csv(latest_file)
        
        return None
    
    def prepare_features(self, raw_data: pd.DataFrame, 
                        feature_cols: list = None) -> Tuple[np.ndarray, list]:
        """
        Prepare features for prediction.
        
        Args:
            raw_data: Raw data DataFrame
            feature_cols: List of feature column names
        
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if feature_cols is None:
            # Default features (from model training)
            feature_cols = [
                'NormClose', 'DailyLogReturn',
                'ALR1W', 'ALR2W', 'ALR1M', 'ALR2M',
                'RSI', 'MACD',
                'VN_Index_Return', 'VN_Index_Volatility',
                'USD_VND_Change', 'Policy_Rate', 'Interbank_Rate',
                'Inflation_YoY',
                'Vol_Lag_1', 'Vol_Lag_5', 'Vol_Lag_10',
                'Vol_MA_20', 'Vol_Std_20',
                'pe_ratio', 'pb_ratio', 'roe', 'roa',
                'debt_to_equity', 'market_cap', 'beta'
            ]
        
        # Extract features
        X = raw_data[feature_cols].fillna(0).values
        
        self.feature_names = feature_cols
        
        return X, feature_cols
    
    def predict_single_stock(self, symbol: str, 
                            lookback_days: int = 30) -> Dict:
        """
        Make predictions for a single stock.
        
        Args:
            symbol: Stock ticker symbol
            lookback_days: Number of historical days to use
        
        Returns:
            Dictionary with predictions and metadata
        """
        data = self.load_latest_data()
        
        if data is None or symbol not in data['Symbol'].values:
            return {'error': f'No data for {symbol}'}
        
        # Get stock data
        stock_data = data[data['Symbol'] == symbol].sort_values('Date').tail(lookback_days)
        
        X, _ = self.prepare_features(stock_data)
        
        # Get latest prediction (last row)
        latest_features = X[-1:] if len(X) > 0 else None
        
        if latest_features is None:
            return {'error': 'Insufficient data'}
        
        # Make predictions
        rf_reg = self.models.get('rf_regressor')
        rf_clf = self.models.get('rf_classifier')
        
        if rf_reg is None or rf_clf is None:
            return {'error': 'Models not loaded'}
        
        volatility_pred = rf_reg.predict(latest_features)[0]
        risk_pred = rf_clf.predict(latest_features)[0]
        risk_proba = rf_clf.predict_proba(latest_features)[0]
        
        risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
        
        return {
            'symbol': symbol,
            'date': stock_data.iloc[-1]['Date'] if 'Date' in stock_data else None,
            'predicted_volatility_5d': float(volatility_pred),
            'predicted_risk_class': int(risk_pred),
            'predicted_risk_name': risk_names[risk_pred],
            'risk_probabilities': {
                'low': float(risk_proba[0]),
                'medium': float(risk_proba[1]),
                'high': float(risk_proba[2])
            },
            'confidence': float(risk_proba.max()),
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def predict_portfolio(self, symbols: list = None,
                         weights: Dict = None) -> Dict:
        """
        Make predictions for a portfolio of stocks.
        
        Args:
            symbols: List of stock symbols
            weights: Dictionary of portfolio weights
        
        Returns:
            Portfolio-level predictions
        """
        data = self.load_latest_data()
        
        if data is None:
            return {'error': 'No data available'}
        
        if symbols is None:
            symbols = data['Symbol'].unique().tolist()[:10]  # Top 10 by default
        
        if weights is None:
            weights = {s: 1/len(symbols) for s in symbols}
        
        # Individual predictions
        individual_predictions = {}
        portfolio_vol = 0
        portfolio_risk_proba = np.zeros(3)
        
        for symbol in symbols:
            pred = self.predict_single_stock(symbol)
            
            if 'error' not in pred:
                individual_predictions[symbol] = pred
                
                w = weights.get(symbol, 0)
                portfolio_vol += w * pred['predicted_volatility_5d']
                
                risk_proba = [
                    pred['risk_probabilities']['low'],
                    pred['risk_probabilities']['medium'],
                    pred['risk_probabilities']['high']
                ]
                portfolio_risk_proba += w * np.array(risk_proba)
        
        # Portfolio risk class
        portfolio_risk_class = np.argmax(portfolio_risk_proba)
        risk_names = ['Low Risk', 'Medium Risk', 'High Risk']
        
        return {
            'portfolio_level': {
                'predicted_volatility_5d': float(portfolio_vol),
                'predicted_risk_class': int(portfolio_risk_class),
                'predicted_risk_name': risk_names[portfolio_risk_class],
                'risk_probabilities': {
                    'low': float(portfolio_risk_proba[0]),
                    'medium': float(portfolio_risk_proba[1]),
                    'high': float(portfolio_risk_proba[2])
                }
            },
            'individual_stocks': individual_predictions,
            'portfolio_composition': weights,
            'prediction_count': len(individual_predictions),
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def generate_report(self, predictions: Dict) -> str:
        """
        Generate human-readable prediction report.
        
        Args:
            predictions: Dictionary from predict_portfolio() or predict_single_stock()
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("VOLATILITY & RISK PREDICTION REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if 'portfolio_level' in predictions:
            # Portfolio report
            portfolio = predictions['portfolio_level']
            report.append("📊 PORTFOLIO SUMMARY")
            report.append("-" * 70)
            report.append(f"Predicted 5-Day Volatility: {portfolio['predicted_volatility_5d']:.6f}")
            report.append(f"Risk Classification: {portfolio['predicted_risk_name']}")
            report.append(f"Risk Probabilities:")
            report.append(f"  - Low:    {portfolio['risk_probabilities']['low']:.1%}")
            report.append(f"  - Medium: {portfolio['risk_probabilities']['medium']:.1%}")
            report.append(f"  - High:   {portfolio['risk_probabilities']['high']:.1%}")
            report.append("")
            
            if predictions['individual_stocks']:
                report.append("📈 INDIVIDUAL STOCKS")
                report.append("-" * 70)
                for symbol, pred in predictions['individual_stocks'].items():
                    report.append(f"{symbol}:")
                    report.append(f"  Volatility: {pred['predicted_volatility_5d']:.6f}")
                    report.append(f"  Risk: {pred['predicted_risk_name']} (conf: {pred['confidence']:.1%})")
        
        else:
            # Single stock report
            report.append("📈 STOCK PREDICTION")
            report.append("-" * 70)
            report.append(f"Symbol: {predictions['symbol']}")
            report.append(f"Predicted 5-Day Volatility: {predictions['predicted_volatility_5d']:.6f}")
            report.append(f"Risk Classification: {predictions['predicted_risk_name']}")
            report.append(f"Confidence: {predictions['confidence']:.1%}")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_predictions(self, predictions: Dict, 
                        output_dir: str = 'data/results/predictions') -> Path:
        """
        Save predictions to disk.
        
        Args:
            predictions: Prediction dictionary
            output_dir: Output directory
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_path / f'predictions_hybrid_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        return filename
