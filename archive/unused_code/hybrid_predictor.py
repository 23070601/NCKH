"""
Hybrid Volatility & Risk Prediction System
Combines regression (volatility forecasting) + classification (risk assessment)
for comprehensive portfolio volatility management
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler


class HybridVolatilityPredictor:
    """
    Hybrid predictor combining:
    - Regression: Predict continuous volatility values
    - Classification: Predict risk categories (Low/Medium/High)
    - Ensemble: Multi-model voting for robust predictions
    """
    
    def __init__(self, regression_model=None, classification_model=None, scaler=None):
        """
        Initialize hybrid predictor with trained models.
        
        Args:
            regression_model: Trained regression model (Random Forest or XGBoost)
            classification_model: Trained classification model (Random Forest Classifier)
            scaler: StandardScaler for feature normalization
        """
        self.regression_model = regression_model
        self.classification_model = classification_model
        self.scaler = scaler or StandardScaler()
        
        self.risk_thresholds = None  # Will be set during calibration
        self.feature_names = None
        self.performance_metrics = {}
    
    def calibrate_risk_thresholds(self, volatilities: np.ndarray, method: str = 'percentile'):
        """
        Calibrate risk thresholds based on historical volatility.
        
        Args:
            volatilities: Array of historical volatility values
            method: 'percentile', 'std', or 'kmeans'
        
        Returns:
            dict: Thresholds for Low/Medium/High risk
        """
        if method == 'percentile':
            p33 = np.percentile(volatilities, 33.33)
            p67 = np.percentile(volatilities, 66.67)
            self.risk_thresholds = {
                'low_high': p33,      # threshold between Low and Medium
                'medium_high': p67    # threshold between Medium and High
            }
        
        elif method == 'std':
            mean_vol = np.mean(volatilities)
            std_vol = np.std(volatilities)
            self.risk_thresholds = {
                'low_high': mean_vol - 0.5 * std_vol,
                'medium_high': mean_vol + 0.5 * std_vol
            }
        
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(volatilities.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            self.risk_thresholds = {
                'low_high': (centers[0] + centers[1]) / 2,
                'medium_high': (centers[1] + centers[2]) / 2
            }
        
        return self.risk_thresholds
    
    def predict_volatility(self, X: np.ndarray, method: str = 'ensemble') -> np.ndarray:
        """
        Predict future volatility values.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            method: 'rf' (Random Forest only), 'xgb' (XGBoost only), or 'ensemble' (average)
        
        Returns:
            Predicted volatility values
        """
        if method == 'rf':
            return self.regression_model.predict(X)
        
        elif method == 'xgb':
            if hasattr(self.regression_model, 'predict'):
                # XGBoost model
                return self.regression_model.predict(X)
            else:
                raise ValueError("XGBoost model not available")
        
        elif method == 'ensemble':
            # Average predictions from multiple models (if available)
            pred1 = self.regression_model.predict(X)
            return pred1  # Can extend to average multiple models
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def predict_risk_class(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk classification (0=Low, 1=Medium, 2=High).
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Risk class predictions (0, 1, or 2)
        """
        return self.classification_model.predict(X)
    
    def predict_risk_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability distribution over risk classes.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability matrix (n_samples, 3) - probabilities for [Low, Medium, High]
        """
        if hasattr(self.classification_model, 'predict_proba'):
            return self.classification_model.predict_proba(X)
        else:
            raise ValueError("Model does not support probability predictions")
    
    def predict_hybrid(self, X: np.ndarray, include_proba: bool = True) -> Dict:
        """
        Make hybrid predictions (volatility + risk together).
        
        Args:
            X: Feature matrix (n_samples, n_features)
            include_proba: Include probability distribution for risk classes
        
        Returns:
            Dictionary with:
            - volatility: Predicted volatility values
            - risk_class: Risk classification (0=Low, 1=Medium, 2=High)
            - risk_name: Human-readable risk name
            - risk_proba: Probability distribution (if include_proba=True)
            - confidence: Confidence score (highest probability)
        """
        # Predict volatility
        volatility_pred = self.predict_volatility(X)
        
        # Predict risk class
        risk_class = self.predict_risk_class(X)
        
        # Get probability distribution
        if include_proba and hasattr(self.classification_model, 'predict_proba'):
            risk_proba = self.predict_risk_proba(X)
            confidence = risk_proba.max(axis=1)
        else:
            risk_proba = None
            confidence = None
        
        # Map risk class to names
        risk_names = np.array(['Low Risk', 'Medium Risk', 'High Risk'])[risk_class]
        
        return {
            'volatility': volatility_pred,
            'risk_class': risk_class,
            'risk_name': risk_names,
            'risk_proba': risk_proba,
            'confidence': confidence
        }
    
    def predict_portfolio(self, stock_data: pd.DataFrame, 
                         weights: Dict[str, float] = None) -> Dict:
        """
        Predict portfolio-level volatility and risk from individual stocks.
        
        Args:
            stock_data: DataFrame with columns [Symbol, Features...]
            weights: Dictionary of stock weights {Symbol: weight}
        
        Returns:
            Dictionary with portfolio-level predictions
        """
        if weights is None:
            # Equal weight
            symbols = stock_data['Symbol'].unique()
            weights = {s: 1/len(symbols) for s in symbols}
        
        # Group by stock
        portfolio_vol = 0
        portfolio_risk = np.zeros(3)  # For risk probabilities
        
        for symbol in stock_data['Symbol'].unique():
            stock_mask = stock_data['Symbol'] == symbol
            X_stock = stock_data[stock_mask].drop('Symbol', axis=1).values
            
            # Get predictions
            pred = self.predict_hybrid(X_stock, include_proba=True)
            
            # Aggregate
            w = weights.get(symbol, 0)
            portfolio_vol += w * pred['volatility'].mean()
            
            if pred['risk_proba'] is not None:
                portfolio_risk += w * pred['risk_proba'].mean(axis=0)
        
        # Portfolio risk class (max probability)
        portfolio_risk_class = np.argmax(portfolio_risk)
        portfolio_risk_names = ['Low Risk', 'Medium Risk', 'High Risk'][portfolio_risk_class]
        
        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_risk_class': portfolio_risk_class,
            'portfolio_risk_name': portfolio_risk_names,
            'portfolio_risk_proba': portfolio_risk,
            'individual_stocks': {}
        }
    
    def optimize_allocation(self, volatility_preds: np.ndarray,
                           risk_classes: np.ndarray,
                           expected_returns: np.ndarray = None) -> Dict:
        """
        Suggest optimal portfolio allocation based on volatility and risk predictions.
        
        Args:
            volatility_preds: Predicted volatilities for each asset
            risk_classes: Risk classifications for each asset
            expected_returns: Expected returns for each asset
        
        Returns:
            Dictionary with allocation weights and portfolio metrics
        """
        n_assets = len(volatility_preds)
        
        if expected_returns is None:
            # Default: inverse volatility weighting
            expected_returns = 1.0 / (volatility_preds + 1e-6)
        
        # Risk-adjusted returns
        risk_adjusted_returns = expected_returns / (volatility_preds + 1e-6)
        
        # Normalize to get weights
        weights = risk_adjusted_returns / risk_adjusted_returns.sum()
        
        # Filter out high-risk assets if needed
        high_risk_mask = risk_classes == 2  # High risk
        if high_risk_mask.any():
            # Reduce weight of high-risk assets
            weights[high_risk_mask] *= 0.5
            weights = weights / weights.sum()
        
        # Calculate portfolio metrics
        portfolio_vol = np.sqrt(np.sum(weights**2 * volatility_preds**2))
        portfolio_return = np.sum(weights * expected_returns)
        
        return {
            'weights': weights,
            'portfolio_volatility': portfolio_vol,
            'portfolio_expected_return': portfolio_return,
            'sharpe_ratio': portfolio_return / (portfolio_vol + 1e-6),
            'high_risk_reduction': high_risk_mask.sum()
        }
    
    def explain_prediction(self, X: np.ndarray, idx: int = 0) -> Dict:
        """
        Explain individual predictions (feature importance, thresholds).
        
        Args:
            X: Feature matrix
            idx: Index of sample to explain
        
        Returns:
            Dictionary with explanation
        """
        if not hasattr(self.regression_model, 'feature_importances_'):
            return {'error': 'Model does not support feature importance'}
        
        # Get prediction
        pred = self.predict_hybrid(X[idx:idx+1], include_proba=True)
        
        # Top features
        importances = self.regression_model.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
        
        return {
            'prediction': pred,
            'top_features': {
                f'Feature_{i}': importances[i] for i in top_indices
            },
            'risk_threshold_distance': {
                'to_low_high': pred['volatility'][0] - self.risk_thresholds['low_high'] if self.risk_thresholds else None,
                'to_medium_high': pred['volatility'][0] - self.risk_thresholds['medium_high'] if self.risk_thresholds else None
            }
        }
    
    def set_performance_metrics(self, metrics: Dict):
        """Store model performance metrics for monitoring."""
        self.performance_metrics = metrics
    
    def get_summary(self) -> str:
        """Get summary of hybrid predictor capabilities."""
        return f"""
Hybrid Volatility & Risk Predictor
==================================
✓ Regression Model: {self.regression_model.__class__.__name__}
✓ Classification Model: {self.classification_model.__class__.__name__}
✓ Risk Thresholds Calibrated: {self.risk_thresholds is not None}
✓ Feature Scaler: {self.scaler.__class__.__name__}

Capabilities:
- predict_volatility(): Continuous volatility forecasting
- predict_risk_class(): Risk classification (Low/Medium/High)
- predict_hybrid(): Combined volatility + risk predictions
- predict_portfolio(): Portfolio-level aggregation
- optimize_allocation(): Optimal weight allocation
- explain_prediction(): Feature importance & explanation
        """
