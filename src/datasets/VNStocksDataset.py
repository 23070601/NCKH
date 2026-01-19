"""
PyTorch Geometric Dataset for Vietnamese FDI Stocks
Adapted from SP100AnalysisWithGNNs for volatility prediction
"""
import os.path as osp
from typing import Callable, Optional

import pandas as pd
import torch
from torch_geometric.data import Dataset, Data


def get_graph_in_pyg_format(values_path: str, adj_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates the PyTorch Geometric graph data from the stock price data and adjacency matrix.
    
    Args:
        values_path: Path of the CSV file containing the stock price data
        adj_path: Path of the NumPy file containing the adjacency matrix
        
    Returns:
        x: Node features (nodes_nb, timestamps_nb, features_nb)
        close_prices: Close prices (nodes_nb, timestamps_nb)
        edge_index: Edge index (2, edge_nb)
        edge_weight: Edge weight (edge_nb,)
    """
    import numpy as np
    
    # Load data
    values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
    adj = np.load(adj_path)
    
    nodes_nb = len(adj)
    edge_nb = np.count_nonzero(adj)
    
    # Extract features (all columns except 'Close')
    feature_cols = [col for col in values.columns if col != 'Close']
    x = torch.tensor(
        values[feature_cols].to_numpy().reshape((nodes_nb, -1, len(feature_cols))),
        dtype=torch.float32
    )  # shape: (nodes_nb, timestamps_nb, features_nb)
    
    # Extract close prices
    close_prices = torch.tensor(
        values['Close'].to_numpy().reshape((nodes_nb, -1)),
        dtype=torch.float32
    )  # shape: (nodes_nb, timestamps_nb)
    
    # Build edge_index and edge_weight from adjacency matrix
    edge_index = torch.zeros((2, edge_nb), dtype=torch.long)
    edge_weight = torch.zeros((edge_nb,), dtype=torch.float32)
    
    count = 0
    for i in range(nodes_nb):
        for j in range(nodes_nb):
            if (weight := adj[i, j]) != 0:
                edge_index[0, count] = i
                edge_index[1, count] = j
                edge_weight[count] = weight
                count += 1
    
    # Swap axes: (nodes_nb, timestamps_nb, features_nb) -> (nodes_nb, features_nb, timestamps_nb)
    x = x.permute(0, 2, 1)
    
    return x, close_prices, edge_index, edge_weight


class VNStocksDataset(Dataset):
    """
    Vietnamese FDI Stock price dataset for PyTorch Geometric.
    
    This dataset loads historical stock data and graph structure (adjacency matrix)
    to create temporal graph snapshots for volatility prediction.
    
    Args:
        root: Root directory where the dataset should be stored
        values_file_name: Name of the CSV file with stock values
        adj_file_name: Name of the NumPy file with adjacency matrix
        past_window: Number of past timesteps to use as input
        future_window: Number of future timesteps to predict
        force_reload: Whether to force reload the dataset
        transform: Optional transform to be applied on a sample
    """
    
    def __init__(
        self,
        root: str = "../data/",
        values_file_name: str = "values.csv",
        adj_file_name: str = "adj.npy",
        past_window: int = 25,  # ~5 weeks (25 trading days)
        future_window: int = 1,
        force_reload: bool = False,
        transform: Optional[Callable] = None
    ):
        self.values_file_name = values_file_name
        self.adj_file_name = adj_file_name
        self.past_window = past_window
        self.future_window = future_window
        super().__init__(root, force_reload=force_reload, transform=transform)
    
    @property
    def raw_file_names(self) -> list[str]:
        """Files that should exist in the raw directory."""
        return [self.values_file_name, self.adj_file_name]
    
    @property
    def processed_file_names(self) -> list[str]:
        """Files to find in the processed directory."""
        return [f'timestep_{idx}.pt' for idx in range(len(self))]
    
    def download(self) -> None:
        """Download the dataset (not needed - data is generated locally)."""
        pass
    
    def process(self) -> None:
        """
        Process raw data files into PyTorch Geometric Data objects.
        
        Creates temporal graph snapshots with:
        - x: Node features for past_window timesteps
        - y: Target values (Close price or volatility) for future_window timesteps
        - edge_index, edge_weight: Graph structure
        - close_price: Historical close prices
        - close_price_y: Future close prices
        """
        # Get graph data in PyTorch Geometric format
        x, close_prices, edge_index, edge_weight = get_graph_in_pyg_format(
            values_path=self.raw_paths[0],
            adj_path=self.raw_paths[1]
        )
        
        # Create temporal snapshots
        # Each snapshot has:
        # - Input: features from t to t+past_window
        # - Target: features at t+past_window to t+past_window+future_window
        timestamps = []
        for idx in range(x.shape[2] - self.past_window - self.future_window):
            data = Data(
                x=x[:, :, idx:idx + self.past_window],  # (nodes, features, past_window)
                edge_index=edge_index,
                edge_weight=edge_weight,
                close_price=close_prices[:, idx:idx + self.past_window],  # Historical prices
                y=x[:, 0, idx + self.past_window:idx + self.past_window + self.future_window],  # Target: NormClose
                close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],  # Future prices
            )
            timestamps.append(data)
        
        # Save processed data
        for t, timestep in enumerate(timestamps):
            torch.save(timestep, osp.join(self.processed_dir, f"timestep_{t}.pt"))
    
    def len(self) -> int:
        """Return the number of temporal snapshots in the dataset."""
        values = pd.read_csv(self.raw_paths[0]).set_index(['Symbol', 'Date'])
        # Get number of timestamps for one stock
        first_symbol = values.index[0][0]
        n_timestamps = len(values.loc[first_symbol])
        return n_timestamps - self.past_window - self.future_window
    
    def get(self, idx: int) -> Data:
        """Get a single temporal snapshot."""
        data = torch.load(osp.join(self.processed_dir, f'timestep_{idx}.pt'))
        return data


class VNStocksVolatilityDataset(VNStocksDataset):
    """
    Vietnamese FDI Stocks Dataset with volatility as target.
    
    Extends VNStocksDataset to predict volatility instead of price.
    Volatility is calculated as rolling standard deviation of returns.
    """
    
    def __init__(
        self,
        root: str = "../data/",
        values_file_name: str = "values.csv",
        adj_file_name: str = "adj.npy",
        past_window: int = 25,
        future_window: int = 5,  # Predict volatility over next 5 days
        volatility_window: int = 20,  # Window for volatility calculation
        force_reload: bool = False,
        transform: Optional[Callable] = None
    ):
        self.volatility_window = volatility_window
        super().__init__(
            root=root,
            values_file_name=values_file_name,
            adj_file_name=adj_file_name,
            past_window=past_window,
            future_window=future_window,
            force_reload=force_reload,
            transform=transform
        )
    
    def process(self) -> None:
        """Process with volatility as target."""
        import numpy as np
        
        # Get graph data
        x, close_prices, edge_index, edge_weight = get_graph_in_pyg_format(
            values_path=self.raw_paths[0],
            adj_path=self.raw_paths[1]
        )
        
        # Calculate returns for volatility computation
        returns = torch.diff(torch.log(close_prices), dim=1)
        
        # Calculate rolling volatility (standard deviation of returns)
        volatility = torch.zeros_like(close_prices)
        for i in range(self.volatility_window, close_prices.shape[1]):
            volatility[:, i] = returns[:, i-self.volatility_window:i].std(dim=1)
        
        # Create temporal snapshots
        timestamps = []
        start_idx = max(self.volatility_window, 0)
        for idx in range(start_idx, x.shape[2] - self.past_window - self.future_window):
            # Calculate future volatility (over future_window period)
            future_vol_start = idx + self.past_window
            future_vol_end = min(future_vol_start + self.future_window, volatility.shape[1])
            future_volatility = volatility[:, future_vol_start:future_vol_end].mean(dim=1, keepdim=True)
            
            data = Data(
                x=x[:, :, idx:idx + self.past_window],
                edge_index=edge_index,
                edge_weight=edge_weight,
                close_price=close_prices[:, idx:idx + self.past_window],
                y=future_volatility,  # Target: volatility
                close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
                volatility=volatility[:, idx:idx + self.past_window],  # Historical volatility
            )
            timestamps.append(data)
        
        # Save processed data
        for t, timestep in enumerate(timestamps):
            torch.save(timestep, osp.join(self.processed_dir, f"timestep_{t}.pt"))
    
    def len(self) -> int:
        """Return the number of temporal snapshots."""
        values = pd.read_csv(self.raw_paths[0]).set_index(['Symbol', 'Date'])
        first_symbol = values.index[0][0]
        n_timestamps = len(values.loc[first_symbol])
        start_idx = max(self.volatility_window, 0)
        return n_timestamps - start_idx - self.past_window - self.future_window
