"""
Advanced Risk Metrics Module
Implements VaR, CVaR, downside risk, and multi-horizon volatility measures
"""

import numpy as np
import pandas as pd
import torch
from typing import Union, Tuple, Optional, Dict
from scipy import stats


class RiskMetrics:
    """
    Calculate advanced risk metrics for stock portfolios and individual stocks.
    
    Metrics included:
    - Historical Volatility (multiple horizons)
    - Value at Risk (VaR)
    - Conditional Value at Risk (CVaR / Expected Shortfall)
    - Downside Risk
    - Sharpe Ratio
    - Maximum Drawdown
    - Beta (systematic risk)
    """
    
    @staticmethod
    def historical_volatility(
        returns: Union[np.ndarray, pd.Series, torch.Tensor],
        windows: list = [5, 20, 60],
        annualize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Calculate rolling historical volatility for multiple time windows.
        
        Args:
            returns: Daily returns (T,) or (N, T) for multiple stocks
            windows: List of window sizes in days
            annualize: Whether to annualize volatility (multiply by sqrt(252))
            
        Returns:
            Dictionary with volatility series for each window
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.cpu().numpy()
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns.shape) == 1:
            returns = returns.reshape(1, -1)
        
        volatilities = {}
        
        for window in windows:
            vol = np.zeros_like(returns)
            for i in range(returns.shape[0]):
                series = pd.Series(returns[i])
                vol[i] = series.rolling(window).std().values
            
            if annualize:
                vol = vol * np.sqrt(252)
            
            volatilities[f'Vol_{window}d'] = vol
        
        return volatilities
    
    @staticmethod
    def value_at_risk(
        returns: Union[np.ndarray, pd.Series, torch.Tensor],
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        VaR represents the maximum loss at a given confidence level.
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level (0.95 = 95%, 0.99 = 99%)
            method: 'historical', 'parametric', or 'cornish-fisher'
            
        Returns:
            VaR value (positive number representing potential loss)
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.cpu().numpy()
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        if method == 'historical':
            # Historical VaR: empirical quantile
            var = -np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # Parametric VaR: assumes normal distribution
            mu = np.mean(returns)
            sigma = np.std(returns)
            var = -(mu + sigma * stats.norm.ppf(1 - confidence_level))
        
        elif method == 'cornish-fisher':
            # Cornish-Fisher VaR: accounts for skewness and kurtosis
            mu = np.mean(returns)
            sigma = np.std(returns)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            z = stats.norm.ppf(1 - confidence_level)
            z_cf = (z + 
                   (z**2 - 1) * skew / 6 +
                   (z**3 - 3*z) * kurt / 24 -
                   (2*z**3 - 5*z) * skew**2 / 36)
            
            var = -(mu + sigma * z_cf)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var
    
    @staticmethod
    def conditional_var(
        returns: Union[np.ndarray, pd.Series, torch.Tensor],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall (ES).
        
        CVaR is the expected loss given that the loss exceeds VaR.
        More conservative than VaR as it considers tail risk.
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level (0.95 = 95%, 0.99 = 99%)
            
        Returns:
            CVaR value (positive number representing expected tail loss)
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.cpu().numpy()
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        # Calculate VaR first
        var = RiskMetrics.value_at_risk(returns, confidence_level, method='historical')
        
        # CVaR is the mean of returns below the VaR threshold
        tail_losses = returns[returns <= -var]
        
        if len(tail_losses) == 0:
            return var  # If no tail losses, return VaR
        
        cvar = -np.mean(tail_losses)
        
        return cvar
    
    @staticmethod
    def downside_risk(
        returns: Union[np.ndarray, pd.Series, torch.Tensor],
        target_return: float = 0.0,
        annualize: bool = True
    ) -> float:
        """
        Calculate downside risk (semi-deviation).
        
        Measures volatility of negative returns only.
        
        Args:
            returns: Daily returns
            target_return: Minimum acceptable return (MAR)
            annualize: Whether to annualize the metric
            
        Returns:
            Downside risk
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.cpu().numpy()
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        # Only consider returns below target
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        if annualize:
            downside_std = downside_std * np.sqrt(252)
        
        return downside_std
    
    @staticmethod
    def sharpe_ratio(
        returns: Union[np.ndarray, pd.Series, torch.Tensor],
        risk_free_rate: float = 0.03,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Measures risk-adjusted return.
        
        Args:
            returns: Daily returns
            risk_free_rate: Annual risk-free rate (default 3%)
            annualize: Whether to annualize
            
        Returns:
            Sharpe ratio
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.cpu().numpy()
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return np.nan
        
        if annualize:
            # Annualize mean and std
            mean_return = mean_return * 252
            std_return = std_return * np.sqrt(252)
        
        sharpe = (mean_return - risk_free_rate) / std_return
        
        return sharpe
    
    @staticmethod
    def sortino_ratio(
        returns: Union[np.ndarray, pd.Series, torch.Tensor],
        target_return: float = 0.0,
        risk_free_rate: float = 0.03,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sortino Ratio.
        
        Similar to Sharpe but uses downside risk instead of total volatility.
        
        Args:
            returns: Daily returns
            target_return: Minimum acceptable return
            risk_free_rate: Annual risk-free rate
            annualize: Whether to annualize
            
        Returns:
            Sortino ratio
        """
        if isinstance(returns, torch.Tensor):
            returns = returns.cpu().numpy()
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        returns = returns.flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return np.nan
        
        mean_return = np.mean(returns)
        downside_std = RiskMetrics.downside_risk(returns, target_return, annualize=False)
        
        if downside_std == 0:
            return np.nan
        
        if annualize:
            mean_return = mean_return * 252
            downside_std = downside_std * np.sqrt(252)
        
        sortino = (mean_return - risk_free_rate) / downside_std
        
        return sortino
    
    @staticmethod
    def maximum_drawdown(
        prices: Union[np.ndarray, pd.Series, torch.Tensor]
    ) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.
        
        Max drawdown is the largest peak-to-trough decline.
        
        Args:
            prices: Price series
            
        Returns:
            Tuple of (max_drawdown, peak_idx, trough_idx)
        """
        if isinstance(prices, torch.Tensor):
            prices = prices.cpu().numpy()
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        prices = prices.flatten()
        prices = prices[~np.isnan(prices)]
        
        if len(prices) == 0:
            return np.nan, 0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = np.min(drawdown)
        trough_idx = np.argmin(drawdown)
        
        # Find peak index (last maximum before trough)
        peak_idx = np.argmax(running_max[:trough_idx + 1])
        
        return max_dd, peak_idx, trough_idx
    
    @staticmethod
    def calculate_beta(
        stock_returns: Union[np.ndarray, pd.Series, torch.Tensor],
        market_returns: Union[np.ndarray, pd.Series, torch.Tensor]
    ) -> float:
        """
        Calculate beta (systematic risk relative to market).
        
        Beta = Cov(stock, market) / Var(market)
        
        Args:
            stock_returns: Stock returns
            market_returns: Market returns (e.g., VN-Index)
            
        Returns:
            Beta coefficient
        """
        if isinstance(stock_returns, torch.Tensor):
            stock_returns = stock_returns.cpu().numpy()
        if isinstance(market_returns, torch.Tensor):
            market_returns = market_returns.cpu().numpy()
        if isinstance(stock_returns, pd.Series):
            stock_returns = stock_returns.values
        if isinstance(market_returns, pd.Series):
            market_returns = market_returns.values
        
        stock_returns = stock_returns.flatten()
        market_returns = market_returns.flatten()
        
        # Remove NaN
        mask = ~(np.isnan(stock_returns) | np.isnan(market_returns))
        stock_returns = stock_returns[mask]
        market_returns = market_returns[mask]
        
        if len(stock_returns) < 2:
            return np.nan
        
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return np.nan
        
        beta = covariance / market_variance
        
        return beta
    
    @staticmethod
    def calculate_all_metrics(
        returns: Union[np.ndarray, pd.Series, torch.Tensor],
        prices: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None,
        market_returns: Optional[Union[np.ndarray, pd.Series, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Calculate all risk metrics for a single stock.
        
        Args:
            returns: Daily returns
            prices: Price series (for max drawdown)
            market_returns: Market returns (for beta)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Volatility
        vols = RiskMetrics.historical_volatility(returns, windows=[5, 20, 60])
        for key, val in vols.items():
            metrics[key] = np.nanmean(val)  # Average over time
        
        # VaR and CVaR
        metrics['VaR_95'] = RiskMetrics.value_at_risk(returns, 0.95)
        metrics['VaR_99'] = RiskMetrics.value_at_risk(returns, 0.99)
        metrics['CVaR_95'] = RiskMetrics.conditional_var(returns, 0.95)
        metrics['CVaR_99'] = RiskMetrics.conditional_var(returns, 0.99)
        
        # Downside risk
        metrics['Downside_Risk'] = RiskMetrics.downside_risk(returns)
        
        # Risk-adjusted returns
        metrics['Sharpe_Ratio'] = RiskMetrics.sharpe_ratio(returns)
        metrics['Sortino_Ratio'] = RiskMetrics.sortino_ratio(returns)
        
        # Maximum drawdown
        if prices is not None:
            max_dd, _, _ = RiskMetrics.maximum_drawdown(prices)
            metrics['Max_Drawdown'] = max_dd
        
        # Beta
        if market_returns is not None:
            metrics['Beta'] = RiskMetrics.calculate_beta(returns, market_returns)
        
        return metrics


def calculate_risk_labels(
    volatility: np.ndarray,
    method: str = 'percentile',
    n_classes: int = 3
) -> np.ndarray:
    """
    Create risk level labels from volatility.
    
    Args:
        volatility: Volatility values
        method: 'percentile', 'threshold', or 'kmeans'
        n_classes: Number of risk classes (default 3: Low, Medium, High)
        
    Returns:
        Risk labels (0 = Low, 1 = Medium, 2 = High, etc.)
    """
    volatility = volatility.flatten()
    volatility = volatility[~np.isnan(volatility)]
    
    if method == 'percentile':
        # Use percentiles to define risk levels
        if n_classes == 3:
            percentiles = [33.33, 66.67]
        elif n_classes == 4:
            percentiles = [25, 50, 75]
        elif n_classes == 5:
            percentiles = [20, 40, 60, 80]
        else:
            percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
        
        thresholds = np.percentile(volatility, percentiles)
        labels = np.digitize(volatility, thresholds)
    
    elif method == 'threshold':
        # Use fixed thresholds (for annualized volatility in %)
        if n_classes == 3:
            thresholds = [0.15, 0.30]  # 15%, 30%
        elif n_classes == 4:
            thresholds = [0.10, 0.20, 0.35]
        else:
            thresholds = np.linspace(0.1, 0.5, n_classes - 1)
        
        labels = np.digitize(volatility, thresholds)
    
    elif method == 'kmeans':
        # Use k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_classes, random_state=42)
        labels = kmeans.fit_predict(volatility.reshape(-1, 1))
        
        # Sort labels by cluster centers (low to high risk)
        centers = kmeans.cluster_centers_.flatten()
        label_order = np.argsort(centers)
        label_mapping = {old: new for new, old in enumerate(label_order)}
        labels = np.array([label_mapping[l] for l in labels])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels


if __name__ == "__main__":
    # Test risk metrics
    print("Testing Risk Metrics Module...")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    n_days = 500
    
    # Simulate returns with fat tails
    returns = np.random.standard_t(df=5, size=n_days) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Simulate market returns
    market_returns = np.random.standard_t(df=5, size=n_days) * 0.015
    
    print("\\nSample Data Generated:")
    print(f"  Returns: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")
    print(f"  Prices: start={prices[0]:.2f}, end={prices[-1]:.2f}")
    
    # Calculate all metrics
    print("\\n" + "="*80)
    print("RISK METRICS")
    print("="*80)
    
    metrics = RiskMetrics.calculate_all_metrics(returns, prices, market_returns)
    
    for metric, value in metrics.items():
        print(f"  {metric:20s}: {value:10.4f}")
    
    # Test risk labels
    print("\\n" + "="*80)
    print("RISK LABELS")
    print("="*80)
    
    volatilities = np.random.lognormal(np.log(0.2), 0.5, 1000)
    labels = calculate_risk_labels(volatilities, method='percentile', n_classes=3)
    
    print(f"  Generated {len(labels)} risk labels")
    print(f"  Distribution: {np.bincount(labels)}")
    print(f"  Labels: Low=0, Medium=1, High=2")
    
    print("\\n✓ Risk metrics module tested successfully!")
