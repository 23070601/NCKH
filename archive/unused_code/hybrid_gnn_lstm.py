"""
Hybrid GNN-LSTM Model for Stock Volatility Prediction
Combines temporal modeling (LSTM/GRU) with graph structure (GNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from typing import Optional, Literal


class TemporalEncoder(nn.Module):
    """
    Encode temporal sequences using LSTM or GRU.
    
    Args:
        in_features: Number of input features per timestep
        hidden_size: Size of hidden state
        num_layers: Number of recurrent layers
        rnn_type: 'lstm' or 'gru'
        dropout: Dropout probability
        bidirectional: Use bidirectional RNN
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        rnn_type: Literal['lstm', 'gru'] = 'lstm',
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(TemporalEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # Choose RNN type
        RNN = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        
        self.rnn = RNN(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal encoder.
        
        Args:
            x: Input (batch*nodes, features, timesteps)
            
        Returns:
            Encoded features (batch*nodes, hidden_size)
        """
        # Transpose to (batch*nodes, timesteps, features)
        x = x.transpose(1, 2)
        
        # RNN forward
        if self.rnn_type == 'lstm':
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward
            h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h_n = h_n[-1]
        
        return h_n


class GraphEncoder(nn.Module):
    """
    Encode graph structure using GNN.
    
    Args:
        in_features: Input feature size
        hidden_size: Hidden layer size
        num_layers: Number of GNN layers
        gnn_type: 'gcn', 'gat', or 'sage'
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        gnn_type: Literal['gcn', 'gat', 'sage'] = 'gcn',
        dropout: float = 0.2,
        num_heads: int = 4
    ):
        super(GraphEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_size
            
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(in_dim, hidden_size))
            elif gnn_type == 'gat':
                # GAT with multi-head attention
                out_dim = hidden_size // num_heads if i < num_layers - 1 else hidden_size
                self.convs.append(GATConv(
                    in_dim,
                    out_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout
                ))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(in_dim, hidden_size))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.norms.append(nn.LayerNorm(hidden_size))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through graph encoder.
        
        Args:
            x: Node features (nodes, features)
            edge_index: Graph connectivity (2, edges)
            edge_weight: Edge weights (edges,)
            
        Returns:
            Encoded features (nodes, hidden_size)
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index, edge_weight)
            x = norm(x)
            
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class HybridGNNLSTM(nn.Module):
    """
    Hybrid model combining temporal (LSTM/GRU) and graph (GNN) encoding.
    
    Architecture:
    1. Temporal Encoder: Process time series per node
    2. Graph Encoder: Aggregate information across nodes
    3. Fusion: Combine temporal and graph features
    4. Predictor: Final output layer
    
    Args:
        in_features: Number of input features per timestep
        hidden_size: Size of hidden representations
        num_temporal_layers: Number of RNN layers
        num_graph_layers: Number of GNN layers
        rnn_type: 'lstm' or 'gru'
        gnn_type: 'gcn', 'gat', or 'sage'
        dropout: Dropout probability
        fusion_method: 'concat', 'add', 'attention'
        output_size: Output dimension (1 for regression)
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_temporal_layers: int = 2,
        num_graph_layers: int = 2,
        rnn_type: Literal['lstm', 'gru'] = 'lstm',
        gnn_type: Literal['gcn', 'gat', 'sage'] = 'gcn',
        dropout: float = 0.2,
        fusion_method: Literal['concat', 'add', 'attention'] = 'concat',
        output_size: int = 1
    ):
        super(HybridGNNLSTM, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            in_features=in_features,
            hidden_size=hidden_size,
            num_layers=num_temporal_layers,
            rnn_type=rnn_type,
            dropout=dropout
        )
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            in_features=self.temporal_encoder.output_size,
            hidden_size=hidden_size,
            num_layers=num_graph_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_size = self.temporal_encoder.output_size + hidden_size
        elif fusion_method == 'add':
            # For addition, temporal and graph features must match
            fusion_size = hidden_size
            if self.temporal_encoder.output_size != hidden_size:
                self.temporal_proj = nn.Linear(self.temporal_encoder.output_size, hidden_size)
        elif fusion_method == 'attention':
            fusion_size = hidden_size
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            if self.temporal_encoder.output_size != hidden_size:
                self.temporal_proj = nn.Linear(self.temporal_encoder.output_size, hidden_size)
        
        # Predictor head
        self.predictor = nn.Sequential(
            nn.Linear(fusion_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch*nodes, features, timesteps) or (nodes, features, timesteps)
            edge_index: Graph connectivity (2, edges)
            edge_weight: Edge weights (edges,)
            
        Returns:
            Predictions (batch*nodes, output_size)
        """
        # Step 1: Temporal encoding
        temporal_features = self.temporal_encoder(x)  # (batch*nodes, hidden)
        
        # Step 2: Graph encoding
        graph_features = self.graph_encoder(
            temporal_features,
            edge_index,
            edge_weight
        )  # (batch*nodes, hidden)
        
        # Step 3: Fusion
        if self.fusion_method == 'concat':
            fused = torch.cat([temporal_features, graph_features], dim=1)
        
        elif self.fusion_method == 'add':
            temp_feat = temporal_features
            if hasattr(self, 'temporal_proj'):
                temp_feat = self.temporal_proj(temp_feat)
            fused = temp_feat + graph_features
        
        elif self.fusion_method == 'attention':
            # Project temporal features if needed
            temp_feat = temporal_features
            if hasattr(self, 'temporal_proj'):
                temp_feat = self.temporal_proj(temp_feat)
            
            # Stack for attention: [temporal, graph]
            stacked = torch.stack([temp_feat, graph_features], dim=1)  # (batch*nodes, 2, hidden)
            
            # Self-attention
            attended, _ = self.attention(stacked, stacked, stacked)
            
            # Pool attention output
            fused = attended.mean(dim=1)  # (batch*nodes, hidden)
        
        # Step 4: Prediction
        output = self.predictor(fused)
        
        return output


class HybridGNNLSTMForPyG(nn.Module):
    """
    Wrapper for PyTorch Geometric Data compatibility.
    
    Handles PyG Data objects and batch processing.
    """
    
    def __init__(self, *args, **kwargs):
        super(HybridGNNLSTMForPyG, self).__init__()
        self.model = HybridGNNLSTM(*args, **kwargs)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass with PyG Data.
        
        Args:
            data: PyG Data object with x, edge_index, edge_weight
            
        Returns:
            Predictions
        """
        return self.model(data.x, data.edge_index, data.edge_weight)


class MultiTaskHybridGNN(nn.Module):
    """
    Multi-task version for joint volatility prediction and risk classification.
    
    Args:
        in_features: Number of input features
        hidden_size: Hidden representation size
        num_risk_classes: Number of risk classes (3 = Low/Medium/High)
        **kwargs: Other arguments for HybridGNNLSTM
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_size: int = 64,
        num_risk_classes: int = 3,
        **kwargs
    ):
        super(MultiTaskHybridGNN, self).__init__()
        
        # Shared encoder
        self.encoder = HybridGNNLSTM(
            in_features=in_features,
            hidden_size=hidden_size,
            output_size=hidden_size,  # Keep as embedding
            **kwargs
        )
        
        # Task-specific heads
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.2)),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.2)),
            nn.Linear(hidden_size // 2, num_risk_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> dict:
        """
        Forward pass with multi-task outputs.
        
        Args:
            x: Input features
            edge_index: Graph connectivity
            edge_weight: Edge weights
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary with 'volatility' and 'risk_class' predictions
        """
        # Get shared embeddings
        embeddings = self.encoder(x, edge_index, edge_weight)
        
        # Task-specific predictions
        volatility = self.volatility_head(embeddings)
        risk_logits = self.risk_head(embeddings)
        
        output = {
            'volatility': volatility,
            'risk_logits': risk_logits,
            'risk_class': torch.argmax(risk_logits, dim=1)
        }
        
        if return_embeddings:
            output['embeddings'] = embeddings
        
        return output


if __name__ == "__main__":
    # Test the hybrid model
    print("Testing Hybrid GNN-LSTM Model...")
    print("="*80)
    
    # Create dummy data
    batch_size = 8
    nodes = 98
    features = 8
    timesteps = 25
    edges = 200
    
    x = torch.randn(batch_size * nodes, features, timesteps)
    edge_index = torch.randint(0, batch_size * nodes, (2, edges))
    edge_weight = torch.rand(edges)
    
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    
    # Test single-task model
    print("\n" + "="*80)
    print("SINGLE-TASK MODEL (Volatility Prediction)")
    print("="*80)
    
    model = HybridGNNLSTM(
        in_features=features,
        hidden_size=64,
        num_temporal_layers=2,
        num_graph_layers=2,
        rnn_type='lstm',
        gnn_type='gcn',
        fusion_method='concat',
        output_size=1
    )
    
    output = model(x, edge_index, edge_weight)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size * nodes}, 1)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test multi-task model
    print("\n" + "="*80)
    print("MULTI-TASK MODEL (Volatility + Risk Classification)")
    print("="*80)
    
    mt_model = MultiTaskHybridGNN(
        in_features=features,
        hidden_size=64,
        num_risk_classes=3,
        num_temporal_layers=2,
        num_graph_layers=2,
        rnn_type='gru',
        gnn_type='gat',
        fusion_method='attention'
    )
    
    mt_output = mt_model(x, edge_index, edge_weight, return_embeddings=True)
    
    print(f"Volatility output shape: {mt_output['volatility'].shape}")
    print(f"Risk logits shape: {mt_output['risk_logits'].shape}")
    print(f"Risk class shape: {mt_output['risk_class'].shape}")
    print(f"Embeddings shape: {mt_output['embeddings'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in mt_model.parameters()):,}")
    
    # Test different fusion methods
    print("\n" + "="*80)
    print("TESTING FUSION METHODS")
    print("="*80)
    
    for fusion in ['concat', 'add', 'attention']:
        model_fusion = HybridGNNLSTM(
            in_features=features,
            hidden_size=64,
            fusion_method=fusion,
            output_size=1
        )
        out = model_fusion(x, edge_index, edge_weight)
        params = sum(p.numel() for p in model_fusion.parameters())
        print(f"  {fusion:10s}: output {out.shape}, params {params:,}")
    
    print("\n✓ Hybrid GNN-LSTM model tested successfully!")
