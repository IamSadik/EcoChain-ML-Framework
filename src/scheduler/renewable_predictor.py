"""
src/scheduler/renewable_predictor.py
"""

import torch
import torch.nn as nn

class RenewablePredictor(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        prediction = self.fc(lstm_out[:, -1, :])  # Use last timestep
        return prediction
    
    def predict(self, node_id, horizon):
        """
        Predict renewable availability for next 'horizon' minutes
        
        Returns: Percentage of renewable energy (0-100)
        """
        # Get historical data for node
        history = self.get_historical_data(node_id, lookback=24)
        
        # Prepare input tensor
        x = torch.tensor(history).unsqueeze(0).float()
        
        # Predict
        with torch.no_grad():
            prediction = self.forward(x)
        
        return prediction.item()