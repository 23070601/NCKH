"""
Training utilities for deep learning models
"""
from datetime import datetime
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import trange


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    device: str = 'cpu',
    task_title: str = "",
    early_stopping_patience: Optional[int] = None
) -> dict:
    """
    Train a PyTorch model with TensorBoard logging.
    
    Args:
        model: Neural network model
        optimizer: Optimizer (Adam, SGD, etc.)
        criterion: Loss function (MSE, MAE, etc.)
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        task_title: Title for TensorBoard run
        early_stopping_patience: Stop if no improvement for N epochs (None to disable)
        
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    writer = SummaryWriter(
        f'runs/{task_title}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{model.__class__.__name__}'
    )
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'best_epoch': 0,
        'best_test_loss': float('inf')
    }
    
    patience_counter = 0
    
    for epoch in (pbar := trange(num_epochs, desc="Training")):
        # Training
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        history['train_loss'].append(train_loss)
        
        # Testing
        test_loss = test_epoch(model, criterion, test_loader, device)
        history['test_loss'].append(test_loss)
        
        # TensorBoard logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        
        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'test_loss': f'{test_loss:.6f}'
        })
        
        # Early stopping check
        if test_loss < history['best_test_loss']:
            history['best_test_loss'] = test_loss
            history['best_epoch'] = epoch
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), f'models/best_{task_title}_{model.__class__.__name__}.pt')
        else:
            patience_counter += 1
        
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"\\nEarly stopping at epoch {epoch}")
            print(f"Best test loss: {history['best_test_loss']:.6f} at epoch {history['best_epoch']}")
            break
    
    writer.close()
    return history


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    device: str = 'cpu'
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        optimizer: Optimizer
        criterion: Loss function
        train_loader: Training data loader
        device: Device to train on
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Handle different model types
        if hasattr(batch, 'edge_index'):
            # Graph-based model
            out = model(batch.x, batch.edge_index, batch.edge_weight)
        else:
            # Non-graph model
            out = model(batch.x)
        
        # Calculate loss
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def test_epoch(
    model: nn.Module,
    criterion: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> float:
    """
    Evaluate model on test set.
    
    Args:
        model: Neural network model
        criterion: Loss function
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Average test loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            if hasattr(batch, 'edge_index'):
                out = model(batch.x, batch.edge_index, batch.edge_weight)
            else:
                out = model(batch.x)
            
            # Calculate loss
            loss = criterion(out, batch.y)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: Neural network model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test training utilities
    from torch_geometric.data import Data
    import os
    
    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 1)
        
        def forward(self, x, edge_index=None, edge_weight=None):
            # x shape: (nodes, features, timesteps)
            return self.fc(x[:, :, -1])  # Use last timestep
    
    # Create dummy dataset
    dummy_data = []
    for i in range(100):
        data = Data(
            x=torch.randn(10, 8, 25),
            y=torch.randn(10, 1),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_weight=torch.rand(20)
        )
        dummy_data.append(data)
    
    # Create data loaders
    train_loader = DataLoader(dummy_data[:80], batch_size=16, shuffle=True)
    test_loader = DataLoader(dummy_data[80:], batch_size=16, shuffle=False)
    
    # Create model
    model = DummyModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Train
    os.makedirs('models', exist_ok=True)
    history = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=5,
        task_title="test",
        early_stopping_patience=3
    )
    
    print(f"\\nTraining complete!")
    print(f"Best test loss: {history['best_test_loss']:.6f}")
    print(f"Best epoch: {history['best_epoch']}")
