"""
ARIMA Model for Volatility Prediction
Classical statistical baseline for time series forecasting
"""
import numpy as np
import torch
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model.
    
    Classical statistical model for time series forecasting.
    Serves as baseline for comparison with ML/DL methods.
    Fits separate ARIMA model for each stock.
    
    Args:
        order: ARIMA order (p, d, q)
               p: AR order (autoregressive)
               d: Integration order (differencing)
               q: MA order (moving average)
        seasonal_order: Seasonal ARIMA order (P, D, Q, s) or None
        enforce_stationarity: Whether to enforce stationarity
        enforce_invertibility: Whether to enforce invertibility
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        
        self.models = []  # One model per stock
        self.is_fitted = False
    
    def _prepare_univariate_series(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract close price or volatility series.
        
        Args:
            x: Input tensor (nodes, features, timesteps)
            
        Returns:
            Time series array (nodes, timesteps)
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        # Use first feature (typically NormClose or volatility)
        if len(x.shape) == 3:
            series = x[:, 0, :]  # (nodes, timesteps)
        elif len(x.shape) == 2:
            series = x  # Already (nodes, timesteps)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        return series
    
    def fit(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Fit ARIMA model for each stock independently.
        
        Args:
            X: Historical data (nodes, features, timesteps)
            y: Not used (ARIMA learns from X only)
        """
        series = self._prepare_univariate_series(X)
        n_stocks = series.shape[0]
        
        self.models = []
        
        for i in range(n_stocks):
            stock_series = series[i, :]
            
            # Handle NaN values
            if np.any(np.isnan(stock_series)):
                # Use simple moving average for missing values
                stock_series = pd.Series(stock_series).fillna(method='ffill').fillna(method='bfill').values
            
            try:
                # Fit ARIMA model
                model = ARIMA(
                    stock_series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility
                )
                fitted_model = model.fit()
                self.models.append(fitted_model)
            
            except Exception as e:
                # If fitting fails, use simpler AR(1) model
                try:
                    model = ARIMA(stock_series, order=(1, 0, 0))
                    fitted_model = model.fit()
                    self.models.append(fitted_model)
                except:
                    # If even that fails, store None (will use mean prediction)
                    self.models.append(None)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: torch.Tensor, steps: int = 1) -> np.ndarray:
        """
        Make predictions for future timesteps.
        
        Args:
            X: Recent historical data (nodes, features, timesteps)
            steps: Number of steps ahead to predict
            
        Returns:
            Predictions (nodes, steps)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        series = self._prepare_univariate_series(X)
        n_stocks = series.shape[0]
        
        predictions = np.zeros((n_stocks, steps))
        
        for i in range(n_stocks):
            if self.models[i] is not None:
                try:
                    # Use the fitted model parameters but forecast from new data
                    stock_series = series[i, :]
                    
                    # Refit with new data (update)
                    model = ARIMA(
                        stock_series,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility
                    )
                    fitted = model.fit()
                    
                    # Forecast
                    forecast = fitted.forecast(steps=steps)
                    predictions[i, :] = forecast
                
                except:
                    # Fallback to last value
                    predictions[i, :] = stock_series[-1]
            else:
                # Use last value if model is None
                predictions[i, :] = series[i, -1]
        
        return predictions
    
    def predict_volatility(self, X: torch.Tensor, volatility_window: int = 20, future_window: int = 5) -> np.ndarray:
        """
        Predict future volatility.
        
        Forecasts future prices, then calculates volatility from forecasted returns.
        
        Args:
            X: Historical data (nodes, features, timesteps)
            volatility_window: Window for volatility calculation
            future_window: How many days ahead to predict volatility for
            
        Returns:
            Predicted volatility (nodes, 1)
        """
        # Predict future prices
        forecast_steps = volatility_window + future_window
        price_forecast = self.predict(X, steps=forecast_steps)
        
        # Calculate returns from forecast
        returns = np.diff(np.log(price_forecast + 1e-8), axis=1)
        
        # Calculate volatility as std of future returns
        volatility = np.zeros((price_forecast.shape[0], 1))
        
        for i in range(price_forecast.shape[0]):
            future_returns = returns[i, -future_window:]
            volatility[i, 0] = np.std(future_returns)
        
        return volatility
    
    def get_params(self):
        """Get model parameters."""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'enforce_stationarity': self.enforce_stationarity,
            'enforce_invertibility': self.enforce_invertibility
        }


if __name__ == "__main__":
    import pandas as pd
    
    # Test the model
    nodes = 98
    features = 8
    timesteps = 100
    
    # Create dummy time series data
    X = torch.randn(nodes, features, timesteps).cumsum(dim=2)  # Cumulative sum to simulate prices
    
    print("Testing ARIMA Model...")
    print(f"Input shape: {X.shape}")
    
    # Create and fit model
    model = ARIMAModel(order=(2, 1, 2))
    print("\\nFitting ARIMA models for each stock...")
    model.fit(X)
    
    # Make predictions
    print("\\nMaking predictions...")
    predictions = model.predict(X, steps=5)
    
    print(f"\\nPredictions shape: {predictions.shape}")
    print(f"Expected: ({nodes}, 5)")
    
    # Test volatility prediction
    print("\\n" + "="*50)
    print("Testing volatility prediction...")
    vol_predictions = model.predict_volatility(X, volatility_window=20, future_window=5)
    
    print(f"Volatility predictions shape: {vol_predictions.shape}")
    print(f"Expected: ({nodes}, 1)")
    
    print(f"\\nVolatility statistics:")
    print(f"  Mean: {vol_predictions.mean():.6f}")
    print(f"  Std:  {vol_predictions.std():.6f}")
    print(f"  Min:  {vol_predictions.min():.6f}")
    print(f"  Max:  {vol_predictions.max():.6f}")
