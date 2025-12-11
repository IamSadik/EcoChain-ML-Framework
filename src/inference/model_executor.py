"""
Model Executor for ML Inference

Handles execution of ML models with energy monitoring and result verification.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExecutor:
    """
    Executes ML inference tasks with energy monitoring.
    
    Supports:
    - Multiple model types (image classification, text, etc.)
    - Compressed and uncompressed models
    - Energy tracking
    - Result hashing for blockchain verification
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize model executor.
        
        Args:
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.models = {}  # Cache of loaded models
        self.execution_history = []
        
        logger.info(f"Initialized ModelExecutor on device: {device}")
    
    def load_model(
        self,
        model_name: str,
        model: nn.Module,
        compressed: bool = False
    ) -> None:
        """
        Load a model into the executor.
        
        Args:
            model_name: Name/ID for the model
            model: PyTorch model
            compressed: Whether this is a compressed model
        """
        model = model.to(self.device)
        model.eval()
        
        key = f"{model_name}_{'compressed' if compressed else 'original'}"
        self.models[key] = model
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Loaded model '{model_name}' "
                   f"({'compressed' if compressed else 'original'}): "
                   f"{num_params:,} parameters")
    
    def execute_inference(
        self,
        model_name: str,
        input_data: torch.Tensor,
        compressed: bool = False,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute ML inference on given input.
        
        Args:
            model_name: Name of the model to use
            input_data: Input tensor
            compressed: Whether to use compressed version
            task_id: Optional task identifier
            
        Returns:
            Dictionary with inference result and metrics
        """
        key = f"{model_name}_{'compressed' if compressed else 'original'}"
        
        if key not in self.models:
            raise ValueError(f"Model '{key}' not loaded!")
        
        model = self.models[key]
        
        # Move input to device
        input_data = input_data.to(self.device)
        
        # Execute inference with timing
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_data)
        
        # Wait for GPU to finish (if using CUDA)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        execution_time = time.time() - start_time
        
        # Process output
        if len(output.shape) > 1:
            # Classification: get predicted class
            predicted_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        else:
            predicted_class = output.item()
            confidence = 1.0
        
        result = {
            'task_id': task_id,
            'model_name': model_name,
            'compressed': compressed,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'execution_time_sec': execution_time,
            'timestamp': time.time()
        }
        
        # Record in history
        self.execution_history.append(result)
        
        logger.debug(f"Inference complete: model={model_name}, "
                    f"compressed={compressed}, time={execution_time:.4f}s")
        
        return result
    
    def execute_batch(
        self,
        model_name: str,
        input_batch: torch.Tensor,
        compressed: bool = False
    ) -> Dict[str, Any]:
        """
        Execute inference on a batch of inputs.
        
        Args:
            model_name: Name of the model
            input_batch: Batch of input tensors (batch_size, ...)
            compressed: Whether to use compressed model
            
        Returns:
            Batch inference results
        """
        key = f"{model_name}_{'compressed' if compressed else 'original'}"
        
        if key not in self.models:
            raise ValueError(f"Model '{key}' not loaded!")
        
        model = self.models[key]
        input_batch = input_batch.to(self.device)
        
        # Execute batch inference
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(input_batch)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        execution_time = time.time() - start_time
        batch_size = input_batch.shape[0]
        
        # Process outputs
        if len(outputs.shape) > 1:
            predicted_classes = outputs.argmax(dim=1).cpu().numpy()
            confidences = torch.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
        else:
            predicted_classes = outputs.cpu().numpy()
            confidences = np.ones_like(predicted_classes)
        
        result = {
            'model_name': model_name,
            'compressed': compressed,
            'batch_size': batch_size,
            'predicted_classes': predicted_classes.tolist(),
            'confidences': confidences.tolist(),
            'total_time_sec': execution_time,
            'time_per_sample_sec': execution_time / batch_size
        }
        
        logger.debug(f"Batch inference complete: {batch_size} samples, "
                    f"time={execution_time:.4f}s, "
                    f"per-sample={result['time_per_sample_sec']:.4f}s")
        
        return result
    
    def get_model_info(self, model_name: str, compressed: bool = False) -> Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            model_name: Name of the model
            compressed: Whether to get info for compressed version
            
        Returns:
            Model information dictionary
        """
        key = f"{model_name}_{'compressed' if compressed else 'original'}"
        
        if key not in self.models:
            raise ValueError(f"Model '{key}' not loaded!")
        
        model = self.models[key]
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_mb = buffer.tell() / (1024 * 1024)
        
        return {
            'model_name': model_name,
            'compressed': compressed,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': size_mb,
            'device': str(self.device)
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about executed inferences.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.execution_history:
            return {}
        
        # Calculate statistics
        total_inferences = len(self.execution_history)
        
        execution_times = [r['execution_time_sec'] for r in self.execution_history]
        avg_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        min_time = np.min(execution_times)
        max_time = np.max(execution_times)
        
        # Count by model and compression
        model_counts = {}
        for record in self.execution_history:
            key = f"{record['model_name']}_{'compressed' if record['compressed'] else 'original'}"
            model_counts[key] = model_counts.get(key, 0) + 1
        
        return {
            'total_inferences': total_inferences,
            'avg_execution_time_sec': avg_time,
            'std_execution_time_sec': std_time,
            'min_execution_time_sec': min_time,
            'max_execution_time_sec': max_time,
            'model_usage_counts': model_counts
        }
    
    def reset_statistics(self) -> None:
        """Reset execution history."""
        self.execution_history = []
        logger.info("Execution statistics reset")


def generate_dummy_input(
    model_type: str,
    batch_size: int = 1
) -> torch.Tensor:
    """
    Generate dummy input for testing.
    
    Args:
        model_type: Type of model ('image', 'text', 'tabular')
        batch_size: Batch size
        
    Returns:
        Dummy input tensor
    """
    if model_type == 'image' or model_type == 'resnet18' or model_type == 'mobilenet':
        # Image classification: (batch, channels, height, width)
        return torch.randn(batch_size, 3, 224, 224)
    
    elif model_type == 'text' or model_type == 'bert':
        # Text: (batch, sequence_length)
        return torch.randint(0, 30000, (batch_size, 128))
    
    elif model_type == 'tabular':
        # Tabular data: (batch, features)
        return torch.randn(batch_size, 100)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class InferenceTask:
    """
    Represents an ML inference task.
    
    Used for scheduling and tracking.
    """
    
    def __init__(
        self,
        task_id: str,
        model_name: str,
        input_data: Any,
        priority: float = 0.5,
        deadline: Optional[float] = None
    ):
        """
        Initialize inference task.
        
        Args:
            task_id: Unique task identifier
            model_name: Name of model to use
            input_data: Input data for inference
            priority: Task priority (0-1, higher is more urgent)
            deadline: Optional deadline (timestamp)
        """
        self.task_id = task_id
        self.model_name = model_name
        self.input_data = input_data
        self.priority = priority
        self.deadline = deadline
        self.created_at = time.time()
        self.completed_at = None
        self.result = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.task_id,
            'model': self.model_name,
            'priority': self.priority,
            'deadline': self.deadline,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'execution_time': self._estimate_execution_time()
        }
    
    def _estimate_execution_time(self) -> float:
        """
        Estimate execution time based on model complexity.
        
        Returns:
            Estimated time in seconds
        """
        # Simple heuristic based on model name
        if 'resnet' in self.model_name.lower():
            return 0.5
        elif 'mobilenet' in self.model_name.lower():
            return 0.3
        elif 'bert' in self.model_name.lower():
            return 0.8
        else:
            return 0.4
    
    def __repr__(self) -> str:
        return f"InferenceTask(id={self.task_id}, model={self.model_name}, priority={self.priority})"
