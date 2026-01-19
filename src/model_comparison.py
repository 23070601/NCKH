"""
Model Comparison Framework for FDI Stock Volatility Prediction
Supporting multiple algorithm groups: Baseline, ML, and DL models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """
    Framework for comparing multiple algorithms for volatility prediction
    
    Algorithm Groups:
    1. Baseline: Historical Mean, ARIMA
    2. ML: Random Forest, XGBoost
    3. DL: LSTM, GRU, CNN-LSTM
    4. DL (opt): Graph Neural Networks (T-GCN, GAT, etc.)
    """
    
    def __init__(self, look_back=20, forecast_horizon=5):
        """
        Initialize model comparator
        
        Args:
            look_back: Number of past timesteps to use as input (window size)
            forecast_horizon: Number of future steps to predict
        """
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        self.results = {}
        self.models = {}
        
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate model performance with multiple metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
        
        Returns:
            Dictionary with all metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'MSE': mse
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def get_comparison_table(self):
        """Get comparison table of all models"""
        df_results = pd.DataFrame(list(self.results.values()))
        return df_results.sort_values('RMSE')
    
    def recommend_best_model(self):
        """Recommend best model based on RMSE"""
        if not self.results:
            return None
        
        best_model = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        return best_model[0], best_model[1]


# ============================================================================
# 1. BASELINE MODELS
# ============================================================================

class HistoricalMeanModel:
    """Simple baseline: use historical mean as prediction"""
    
    def __init__(self, window=20):
        self.window = window
        self.train_mean = None
    
    def fit(self, y_train):
        """Fit model on training data"""
        self.train_mean = np.mean(y_train)
        return self
    
    def predict(self, X_test, y_test=None):
        """Predict using historical mean"""
        return np.full(len(X_test), self.train_mean)


class ARIMAModel:
    """ARIMA baseline model"""
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model
        
        Args:
            order: (p, d, q) tuple for ARIMA parameters
        """
        self.order = order
        self.model = None
    
    def fit(self, y_train):
        """Fit ARIMA model"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(y_train, order=self.order).fit()
        except ImportError:
            print("Install statsmodels: pip install statsmodels")
        return self
    
    def predict(self, steps):
        """Forecast with ARIMA"""
        if self.model is None:
            raise ValueError("Model not fitted")
        forecast = self.model.get_forecast(steps=steps)
        return forecast.predicted_mean.values


# ============================================================================
# 2. MACHINE LEARNING MODELS
# ============================================================================

class RandomForestVolatilityModel:
    """Random Forest for volatility prediction"""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train):
        """Fit Random Forest model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


class XGBoostVolatilityModel:
    """XGBoost for volatility prediction"""
    
    def __init__(self, max_depth=6, learning_rate=0.1, n_estimators=100):
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=42
            )
        except ImportError:
            print("Install xgboost: pip install xgboost")
        self.scaler = StandardScaler()
    
    def fit(self, X_train, y_train):
        """Fit XGBoost model"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


# ============================================================================
# 3. DEEP LEARNING MODELS
# ============================================================================

class LSTMVolatilityModel:
    """LSTM model for time-series volatility prediction"""
    
    def __init__(self, lookback=20, lstm_units=50, epochs=100, batch_size=32):
        """
        Initialize LSTM model
        
        Args:
            lookback: Number of previous timesteps
            lstm_units: Number of LSTM units
            epochs: Training epochs
            batch_size: Batch size for training
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def _build_model(self, input_shape):
        """Build LSTM architecture"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            self.model = Sequential([
                LSTM(self.lstm_units, input_shape=input_shape, 
                     return_sequences=True),
                Dropout(0.2),
                LSTM(self.lstm_units, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        except ImportError:
            print("Install TensorFlow: pip install tensorflow")
    
    def fit(self, X_train, y_train):
        """Fit LSTM model"""
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        
        if self.model is None:
            self._build_model((X_train.shape[1], X_train.shape[2]) if len(X_train.shape) > 1 
                            else (self.lookback, 1))
        
        try:
            self.history = self.model.fit(
                X_train_scaled, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                validation_split=0.2
            )
        except Exception as e:
            print(f"Training error: {e}")
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        return self.model.predict(X_test_scaled, verbose=0)


class GRUVolatilityModel:
    """GRU model for time-series volatility prediction"""
    
    def __init__(self, lookback=20, gru_units=50, epochs=100, batch_size=32):
        """
        Initialize GRU model
        
        Args:
            lookback: Number of previous timesteps
            gru_units: Number of GRU units
            epochs: Training epochs
            batch_size: Batch size for training
        """
        self.lookback = lookback
        self.gru_units = gru_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
    
    def _build_model(self, input_shape):
        """Build GRU architecture"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import GRU, Dense, Dropout
            
            self.model = Sequential([
                GRU(self.gru_units, input_shape=input_shape, 
                    return_sequences=True),
                Dropout(0.2),
                GRU(self.gru_units, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        except ImportError:
            print("Install TensorFlow: pip install tensorflow")
    
    def fit(self, X_train, y_train):
        """Fit GRU model"""
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        
        if self.model is None:
            self._build_model((X_train.shape[1], X_train.shape[2]) if len(X_train.shape) > 1 
                            else (self.lookback, 1))
        
        try:
            self.history = self.model.fit(
                X_train_scaled, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                validation_split=0.2
            )
        except Exception as e:
            print(f"Training error: {e}")
        return self
    
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
        return self.model.predict(X_test_scaled, verbose=0)


# ============================================================================
# 4. ADVANCED DL MODELS (Optional)
# ============================================================================

class TemporalGCNVolatilityModel:
    """
    Temporal Graph Convolutional Network (T-GCN) for GNN-based prediction
    Advanced DL model leveraging stock correlation graphs
    """
    
    def __init__(self, gnn_type='T-GCN', temporal_units=50, graph_units=50):
        """
        Initialize T-GCN model
        
        Args:
            gnn_type: Type of GNN ('T-GCN', 'STGCN', 'GAT')
            temporal_units: Units in temporal component
            graph_units: Units in graph component
        """
        self.gnn_type = gnn_type
        self.temporal_units = temporal_units
        self.graph_units = graph_units
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X_train, adjacency_matrix, y_train):
        """
        Fit T-GCN model
        
        Args:
            X_train: Node features (stocks x features x time)
            adjacency_matrix: Graph adjacency matrix (stocks x stocks)
            y_train: Target labels
        """
        print(f"T-GCN model initialized (type: {self.gnn_type})")
        print(f"  Temporal units: {self.temporal_units}")
        print(f"  Graph units: {self.graph_units}")
        print(f"  Graph shape: {adjacency_matrix.shape}")
        # Implementation depends on PyTorch Geometric
        # This is a placeholder for the framework
        return self
    
    def predict(self, X_test, adjacency_matrix):
        """Make predictions using GNN"""
        print(f"T-GCN predictions (using {self.gnn_type})")
        return np.zeros(len(X_test))  # Placeholder


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_lag_features(data, lags=[1, 5, 10, 20]):
    """
    Create lag features for ML models
    
    Args:
        data: Time series data
        lags: List of lag values to create
    
    Returns:
        DataFrame with lag and rolling statistics features
    """
    df = pd.DataFrame()
    df['target'] = data
    
    for lag in lags:
        df[f'lag_{lag}'] = pd.Series(data).shift(lag)
    
    # Rolling statistics
    df['rolling_mean_5'] = pd.Series(data).rolling(5).mean()
    df['rolling_std_5'] = pd.Series(data).rolling(5).std()
    df['rolling_mean_10'] = pd.Series(data).rolling(10).mean()
    df['rolling_std_10'] = pd.Series(data).rolling(10).std()
    
    return df.dropna()


def create_sequences(data, lookback):
    """
    Create sequences for RNN models
    
    Args:
        data: Time series data
        lookback: Length of input sequences
    
    Returns:
        X, y sequences
    """
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)


def normalize_data(data):
    """Normalize data to [0, 1]"""
    scaler = StandardScaler()
    return scaler.fit_transform(data), scaler


def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
    """Split data into train, validation, test sets"""
    n = len(X)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    print("Model Comparison Framework Loaded")
    print("\nSupported Models:")
    print("  Baseline: HistoricalMean, ARIMA")
    print("  ML: RandomForest, XGBoost")
    print("  DL: LSTM, GRU")
    print("  DL (opt): T-GCN, GAT, STGCN")
