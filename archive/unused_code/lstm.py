"""
LSTM Model for Volatility Prediction
Handles temporal dependencies in stock price data
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data


class LSTMModel(nn.Module):
    """
    LSTM-based model for stock volatility prediction.
    
    Processes temporal sequences of stock features to predict future volatility.
    Suitable for capturing long-term dependencies in time series data.
    
    Args:
        in_features: Number of input features per timestep
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(LSTMModel, self).__init__()
        
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, nodes, features, timesteps)
               or (nodes, features, timesteps) for single sample
               
        Returns:
            Predictions of shape (batch * nodes, 1) or (nodes, 1)
        """
        # Handle PyG Data object
        if isinstance(x, Data):
            x = x.x
        
        # Reshape: (batch*nodes, features, timesteps) -> (batch*nodes, timesteps, features)
        if len(x.shape) == 4:
            batch, nodes, features, timesteps = x.shape
            x = x.view(batch * nodes, features, timesteps)
        elif len(x.shape) == 3:
            nodes, features, timesteps = x.shape
            batch = 1
            x = x.view(nodes, features, timesteps)
        
        # Transpose to (batch*nodes, timesteps, features)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_n = h_n[-1]
        
        # Final prediction
        out = self.fc(h_n)
        
        return out


class LSTMModelForPyG(nn.Module):
    """
    LSTM Model adapted for PyTorch Geometric graphs.
    
    Processes node features independently using LSTM, then optionally
    aggregates using graph structure.
    
    Args:
        in_features: Number of input features per timestep
        hidden_size: Size of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        use_graph: Whether to use graph structure for aggregation
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_graph: bool = False
    ):
        super(LSTMModelForPyG, self).__init__()
        
        self.use_graph = use_graph
        
        # LSTM for temporal processing
        self.lstm = LSTMModel(
            in_features=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Optional graph aggregation
        if use_graph:
            from torch_geometric.nn import GCNConv
            self.graph_conv = GCNConv(1, 1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional graph aggregation.
        
        Args:
            x: Node features (nodes, features, timesteps)
            edge_index: Graph connectivity (optional)
            edge_weight: Edge weights (optional)
            
        Returns:
            Predictions (nodes, 1)
        """
        # LSTM processing
        out = self.lstm(x)
        
        # Optional graph aggregation
        if self.use_graph and edge_index is not None:
            out = self.graph_conv(out, edge_index, edge_weight)
        
        return out


if __name__ == "__main__":
    # Test the model
    batch_size = 8
    nodes = 98
    features = 8
    timesteps = 25
    
    # Create dummy data
    x = torch.randn(batch_size, nodes, features, timesteps)
    
    # Create model
    model = LSTMModel(in_features=features, hidden_size=64, num_layers=2)
    
    # Forward pass
    out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: ({batch_size * nodes}, 1)")
    
    # Test PyG version
    print("\\n" + "="*50)
    print("Testing PyG version...")
    
    x_single = torch.randn(nodes, features, timesteps)
    edge_index = torch.randint(0, nodes, (2, 100))
    edge_weight = torch.rand(100)
    
    model_pyg = LSTMModelForPyG(in_features=features, use_graph=True)
    out_pyg = model_pyg(x_single, edge_index, edge_weight)
    
    print(f"Input shape: {x_single.shape}")
    print(f"Output shape: {out_pyg.shape}")
    print(f"Expected: ({nodes}, 1)")
