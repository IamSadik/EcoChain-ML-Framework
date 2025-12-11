"""
Renewable Energy Predictor using LSTM

This module implements an LSTM-based neural network for predicting
renewable energy availability for edge nodes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RenewableEnergyLSTM(nn.Module):
    """
    LSTM model for renewable energy prediction.
    
    Predicts future renewable energy availability based on historical patterns.
    """
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features (time, solar_power, wind_power, etc.)
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(RenewableEnergyLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)
        
        # Activation (sigmoid to output 0-1 for percentage)
        self.sigmoid = nn.Sigmoid()
        
        logger.info(f"Initialized LSTM predictor: "
                   f"input={input_size}, hidden={hidden_size}, layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layer
        output = self.fc(last_output)
        
        # Sigmoid activation to get percentage (0-1)
        output = self.sigmoid(output)
        
        return output


class RenewablePredictor:
    """
    High-level renewable energy prediction interface.
    
    Handles data preprocessing, model training, and prediction.
    """
    
    def __init__(
        self,
        lookback_hours: int = 24,
        prediction_horizon_hours: int = 1,
        device: str = 'cpu'
    ):
        """
        Initialize predictor.
        
        Args:
            lookback_hours: Number of past hours to use for prediction
            prediction_horizon_hours: How far ahead to predict
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.lookback_hours = lookback_hours
        self.prediction_horizon = prediction_horizon_hours
        self.device = torch.device(device)
        
        # Initialize model
        self.model = RenewableEnergyLSTM(
            input_size=4,  # hour, day_of_week, solar_power, wind_power
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        # Training flag
        self.is_trained = False
        
        # Normalization parameters (will be set during training)
        self.mean = None
        self.std = None
        
        logger.info(f"Initialized RenewablePredictor: "
                   f"lookback={lookback_hours}h, horizon={prediction_horizon_hours}h")
    
    def prepare_data(
        self,
        time_series: np.ndarray,
        lookback: int,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series data for LSTM training.
        
        Args:
            time_series: Array of shape (n_samples, n_features)
            lookback: Number of past time steps to use
            horizon: Number of future time steps to predict
            
        Returns:
            Tuple of (X, y) where X is input sequences and y is targets
        """
        X, y = [], []
        
        for i in range(len(time_series) - lookback - horizon + 1):
            # Input sequence
            X.append(time_series[i:i+lookback])
            # Target (renewable percentage at future time)
            y.append(time_series[i+lookback+horizon-1, -1])  # Last feature is target
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        historical_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model on historical data.
        
        Args:
            historical_data: Array of shape (n_samples, n_features)
                Features: [hour, day_of_week, solar_power, wind_power]
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training history
        """
        logger.info("Starting model training...")
        
        # Normalize data
        self.mean = historical_data.mean(axis=0)
        self.std = historical_data.std(axis=0) + 1e-8
        normalized_data = (historical_data - self.mean) / self.std
        
        # Prepare sequences
        X, y = self.prepare_data(
            normalized_data,
            lookback=self.lookback_hours,
            horizon=self.prediction_horizon
        )
        
        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            total_train_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / (len(X_train) / batch_size)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = criterion(val_predictions, y_val).item()
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"Train Loss = {avg_train_loss:.4f}, "
                           f"Val Loss = {val_loss:.4f}")
        
        self.is_trained = True
        logger.info("Training completed successfully!")
        
        return history
    
    def predict(
        self,
        recent_data: np.ndarray,
        node_id: Optional[str] = None
    ) -> float:
        """
        Predict renewable energy availability.
        
        Args:
            recent_data: Recent historical data of shape (lookback_hours, n_features)
            node_id: Optional node identifier for logging
            
        Returns:
            Predicted renewable energy percentage (0-100)
        """
        if not self.is_trained:
            # If not trained, use simple heuristic
            logger.warning("Model not trained, using heuristic prediction")
            return self._heuristic_prediction(recent_data)
        
        self.model.eval()
        
        # Normalize input
        normalized = (recent_data - self.mean) / self.std
        
        # Convert to tensor
        x = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(x)
        
        # Convert to percentage (0-100)
        percentage = prediction.item() * 100
        
        logger.debug(f"Predicted renewable: {percentage:.1f}% "
                    f"{'for ' + node_id if node_id else ''}")
        
        return percentage
    
    def _heuristic_prediction(self, recent_data: np.ndarray) -> float:
        """
        Simple heuristic prediction when model is not trained.
        
        Uses recent average as prediction.
        """
        # Assume last column is renewable percentage
        recent_avg = recent_data[:, -1].mean() * 100
        return np.clip(recent_avg, 0, 100)
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'mean': self.mean,
            'std': self.std,
            'lookback_hours': self.lookback_hours,
            'prediction_horizon': self.prediction_horizon
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.lookback_hours = checkpoint['lookback_hours']
        self.prediction_horizon = checkpoint['prediction_horizon']
        self.is_trained = True
        
        logger.info(f"Model loaded from {path}")


def generate_synthetic_renewable_data(
    hours: int = 720,  # 30 days
    solar_capacity: float = 1000,
    wind_capacity: float = 800
) -> np.ndarray:
    """
    Generate synthetic renewable energy data for training.
    
    Args:
        hours: Number of hours of data to generate
        solar_capacity: Solar panel capacity in Watts
        wind_capacity: Wind turbine capacity in Watts
        
    Returns:
        Array of shape (hours, 4) with [hour, day_of_week, solar_power, wind_power]
    """
    data = []
    
    for h in range(hours):
        hour_of_day = h % 24
        day_of_week = (h // 24) % 7
        
        # Solar power (sinusoidal, only during day)
        if 6 <= hour_of_day <= 18:
            solar_factor = np.sin((hour_of_day - 6) * np.pi / 12)
            solar_power = solar_capacity * solar_factor * np.random.uniform(0.8, 1.2)
        else:
            solar_power = 0.0
        
        # Wind power (stochastic, 24/7)
        wind_power = wind_capacity * 0.35 * np.random.uniform(0.3, 1.5)
        
        # Normalize to 0-1 for training
        data.append([
            hour_of_day / 24.0,
            day_of_week / 7.0,
            solar_power / solar_capacity,
            wind_power / wind_capacity
        ])
    
    return np.array(data)
