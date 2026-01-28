"""
Enhanced PyTorch Geometric Dataset with Macro and Fundamental Features
Extends VNStocksDataset to include additional features for comprehensive risk modeling
"""
import os.path as osp
from typing import Callable, Optional

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, Data

from .VNStocksDataset import get_graph_in_pyg_format


class EnhancedVNStocksDataset(Dataset):
    """
    Enhanced Vietnamese FDI Stock dataset with:
    - Original technical indicators
    - Macroeconomic features (VN-Index, rates, inflation, FX)
    - Fundamental features (P/E, ROE, debt ratios, etc.)
    - Advanced risk metrics (VaR, CVaR, downside risk)
    
    Args:
        root: Root directory where the dataset should be stored
        values_file_name: Name of the CSV file with stock values
        adj_file_name: Name of the NumPy file with adjacency matrix
        macro_file_name: Name of the CSV file with macro data
        fundamentals_file_name: Name of the CSV file with fundamentals
        past_window: Number of past timesteps to use as input
        future_window: Number of future timesteps to predict
        include_macro: Whether to include macroeconomic features
        include_fundamentals: Whether to include fundamental features
        force_reload: Whether to force reload the dataset
        transform: Optional transform to be applied on a sample
    """
    
    def __init__(
        self,
        root: str = "../data/",
        values_file_name: str = "values_enriched.csv",
        adj_file_name: str = "adj.npy",
        macro_file_name: str = "macro_data.csv",
        fundamentals_file_name: str = "fundamentals.csv",
        past_window: int = 25,
        future_window: int = 1,
        include_macro: bool = True,
        include_fundamentals: bool = True,
        force_reload: bool = False,
        transform: Optional[Callable] = None
    ):
        self.values_file_name = values_file_name
        self.adj_file_name = adj_file_name
        self.macro_file_name = macro_file_name
        self.fundamentals_file_name = fundamentals_file_name
        self.past_window = past_window
        self.future_window = future_window
        self.include_macro = include_macro
        self.include_fundamentals = include_fundamentals
        
        super().__init__(root, force_reload=force_reload, transform=transform)
    
    @property
    def raw_file_names(self) -> list[str]:
        """Files that should exist in the raw directory."""
        files = [self.values_file_name, self.adj_file_name]
        
        if self.include_macro:
            files.append(self.macro_file_name)
        if self.include_fundamentals:
            files.append(self.fundamentals_file_name)
        
        return files
    
    @property
    def processed_file_names(self) -> list[str]:
        """Files to find in the processed directory."""
        return [f'enhanced_timestep_{idx}.pt' for idx in range(len(self))]
    
    def download(self) -> None:
        """Download the dataset (not needed - data is generated locally)."""
        pass
    
    def _get_fundamental_features(self) -> Optional[torch.Tensor]:
        """
        Load fundamental features as static node features.
        
        Returns:
            Tensor of shape (nodes, num_fundamental_features)
        """
        if not self.include_fundamentals:
            return None
        
        fund_path = osp.join(self.raw_dir, self.fundamentals_file_name)
        
        if not osp.exists(fund_path):
            print(f"Warning: Fundamentals file not found at {fund_path}")
            return None
        
        fundamentals = pd.read_csv(fund_path).set_index('ticker')
        
        # Select numerical features (exclude ticker)
        numeric_cols = fundamentals.select_dtypes(include=[np.number]).columns
        fund_features = fundamentals[numeric_cols].values
        
        # Handle NaN values
        fund_features = np.nan_to_num(fund_features, nan=0.0)
        
        return torch.tensor(fund_features, dtype=torch.float32)
    
    def process(self) -> None:
        """
        Process raw data files into PyTorch Geometric Data objects.
        
        Creates temporal graph snapshots with enhanced features:
        - x: Node features (technical + macro + fundamentals)
        - y: Target values (volatility or returns)
        - edge_index, edge_weight: Graph structure
        - Additional metadata
        """
        # Get base graph data (technical indicators)
        values_path = osp.join(self.raw_dir, self.values_file_name)
        adj_path = osp.join(self.raw_dir, self.adj_file_name)
        
        # Check if enriched file exists, otherwise use base file
        if not osp.exists(values_path):
            print(f"Warning: {values_path} not found, trying base values.csv")
            values_path = osp.join(self.raw_dir, "values.csv")
        
        x, close_prices, edge_index, edge_weight = get_graph_in_pyg_format(
            values_path=values_path,
            adj_path=adj_path
        )
        
        # Get fundamental features (static per node)
        fundamental_features = self._get_fundamental_features()
        
        # Create temporal snapshots
        timestamps = []
        for idx in range(x.shape[2] - self.past_window - self.future_window):
            # Base features
            node_features = x[:, :, idx:idx + self.past_window]  # (nodes, features, past_window)
            
            # Add fundamental features if available
            # Fundamentals are static, so we broadcast across timesteps
            if fundamental_features is not None:
                # Expand fundamentals to match timesteps
                fund_expanded = fundamental_features.unsqueeze(-1).expand(
                    -1, -1, self.past_window
                )  # (nodes, fund_features, past_window)
                
                # Concatenate with temporal features
                node_features = torch.cat([node_features, fund_expanded], dim=1)
            
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_weight=edge_weight,
                close_price=close_prices[:, idx:idx + self.past_window],
                y=x[:, 0, idx + self.past_window:idx + self.past_window + self.future_window],
                close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
                # Store fundamental features separately for reference
                fundamentals=fundamental_features if fundamental_features is not None else torch.tensor([])
            )
            timestamps.append(data)
        
        # Save processed data
        for t, timestep in enumerate(timestamps):
            torch.save(timestep, osp.join(self.processed_dir, f"enhanced_timestep_{t}.pt"))
    
    def len(self) -> int:
        """Return the number of temporal snapshots in the dataset."""
        values_path = osp.join(self.raw_dir, self.values_file_name)
        
        if not osp.exists(values_path):
            values_path = osp.join(self.raw_dir, "values.csv")
        
        values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
        first_symbol = values.index[0][0]
        n_timestamps = len(values.loc[first_symbol])
        return n_timestamps - self.past_window - self.future_window
    
    def get(self, idx: int) -> Data:
        """Get a single temporal snapshot."""
        data = torch.load(
            osp.join(self.processed_dir, f'enhanced_timestep_{idx}.pt'),
            weights_only=False
        )
        return data


class EnhancedVolatilityDataset(EnhancedVNStocksDataset):
    """
    Enhanced dataset for volatility prediction with risk metrics.
    
    Extends EnhancedVNStocksDataset to predict:
    - Future volatility
    - Value at Risk (VaR)
    - Conditional VaR (CVaR)
    - Risk classification labels
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        calculate_var: bool = True,
        var_confidence: float = 0.95,
        risk_labels: bool = True,
        num_risk_classes: int = 3,
        **kwargs
    ):
        self.volatility_window = volatility_window
        self.calculate_var = calculate_var
        self.var_confidence = var_confidence
        self.risk_labels = risk_labels
        self.num_risk_classes = num_risk_classes
        
        super().__init__(**kwargs)
    
    def process(self) -> None:
        """Process with volatility and risk metrics as targets."""
        from ..risk_metrics import RiskMetrics, calculate_risk_labels
        
        # Get base graph data
        values_path = osp.join(self.raw_dir, self.values_file_name)
        if not osp.exists(values_path):
            values_path = osp.join(self.raw_dir, "values.csv")
        
        adj_path = osp.join(self.raw_dir, self.adj_file_name)
        
        x, close_prices, edge_index, edge_weight = get_graph_in_pyg_format(
            values_path=values_path,
            adj_path=adj_path
        )
        
        # Calculate returns for risk metrics
        returns = torch.diff(torch.log(close_prices + 1e-8), dim=1)
        
        # Calculate rolling volatility
        volatility = torch.zeros_like(close_prices)
        for i in range(self.volatility_window, close_prices.shape[1]):
            volatility[:, i] = returns[:, i-self.volatility_window:i].std(dim=1)
        
        # Calculate VaR and CVaR if requested
        var_values = None
        cvar_values = None
        
        if self.calculate_var:
            var_values = torch.zeros_like(close_prices)
            cvar_values = torch.zeros_like(close_prices)
            
            for i in range(self.volatility_window, close_prices.shape[1]):
                for stock_idx in range(close_prices.shape[0]):
                    stock_returns = returns[stock_idx, i-self.volatility_window:i].numpy()
                    var_values[stock_idx, i] = RiskMetrics.value_at_risk(
                        stock_returns, self.var_confidence
                    )
                    cvar_values[stock_idx, i] = RiskMetrics.conditional_var(
                        stock_returns, self.var_confidence
                    )
        
        # Get fundamental features
        fundamental_features = self._get_fundamental_features()
        
        # Create temporal snapshots
        timestamps = []
        start_idx = max(self.volatility_window, 0)
        
        for idx in range(start_idx, x.shape[2] - self.past_window - self.future_window):
            # Calculate future volatility
            future_vol_start = idx + self.past_window
            future_vol_end = min(future_vol_start + self.future_window, volatility.shape[1])
            future_volatility = volatility[:, future_vol_start:future_vol_end].mean(dim=1, keepdim=True)
            
            # Base features
            node_features = x[:, :, idx:idx + self.past_window]
            
            # Add fundamentals if available
            if fundamental_features is not None:
                fund_expanded = fundamental_features.unsqueeze(-1).expand(-1, -1, self.past_window)
                node_features = torch.cat([node_features, fund_expanded], dim=1)
            
            # Prepare target dictionary
            targets = {
                'volatility': future_volatility
            }
            
            # Add VaR/CVaR if calculated
            if self.calculate_var and var_values is not None:
                future_var = var_values[:, future_vol_start:future_vol_end].mean(dim=1, keepdim=True)
                future_cvar = cvar_values[:, future_vol_start:future_vol_end].mean(dim=1, keepdim=True)
                targets['var'] = future_var
                targets['cvar'] = future_cvar
            
            # Add risk labels if requested
            if self.risk_labels:
                risk_classes = calculate_risk_labels(
                    future_volatility.numpy(),
                    method='percentile',
                    n_classes=self.num_risk_classes
                )
                targets['risk_class'] = torch.tensor(risk_classes, dtype=torch.long)
            
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_weight=edge_weight,
                close_price=close_prices[:, idx:idx + self.past_window],
                y=future_volatility,  # Primary target
                close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
                volatility=volatility[:, idx:idx + self.past_window],
                fundamentals=fundamental_features if fundamental_features is not None else torch.tensor([]),
                # Additional targets
                **targets
            )
            timestamps.append(data)
        
        # Save processed data
        for t, timestep in enumerate(timestamps):
            torch.save(timestep, osp.join(self.processed_dir, f"enhanced_timestep_{t}.pt"))
    
    def len(self) -> int:
        """Return the number of temporal snapshots."""
        values_path = osp.join(self.raw_dir, self.values_file_name)
        if not osp.exists(values_path):
            values_path = osp.join(self.raw_dir, "values.csv")
        
        values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
        first_symbol = values.index[0][0]
        n_timestamps = len(values.loc[first_symbol])
        start_idx = max(self.volatility_window, 0)
        return n_timestamps - start_idx - self.past_window - self.future_window
