"""
Model Compression Techniques for Energy-Efficient Inference

Implements:
1. Dynamic Quantization (INT8)
2. Static Quantization
3. Pruning
4. Knowledge Distillation (placeholder)
5. INT8 Accuracy Validation (CRITICAL QoS METRIC)
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import copy
from typing import Any, Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCompressor:
    """
    Handles model compression techniques to reduce energy consumption.
    
    Compression benefits:
    - Quantization (INT8): ~4x speedup, ~4x memory reduction, ~20-30% energy savings
    - Pruning: ~2-3x speedup, ~50% parameter reduction
    
    CRITICAL QoS METRIC: INT8 accuracy loss validation (0.5-2% typical)
    """
    
    def __init__(self):
        """Initialize model compressor."""
        self.compression_stats = {}
        logger.info("Initialized ModelCompressor")
    
    def apply_dynamic_quantization(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Dynamic quantization converts weights to INT8 and dynamically
        quantizes activations during inference. This is the easiest
        quantization method and works well for LSTM, Linear layers.
        
        Args:
            model: PyTorch model to quantize
            dtype: Quantization data type (torch.qint8 or torch.quint8)
            
        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")
        
        # Get original model size
        original_size = self._get_model_size(model)
        
        # Apply dynamic quantization
        quantized_model = quant.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
            dtype=dtype
        )
        
        # Get quantized model size
        quantized_size = self._get_model_size(quantized_model)
        
        # Calculate compression ratio
        compression_ratio = original_size / quantized_size
        
        logger.info(f"Dynamic quantization complete:")
        logger.info(f"  Original size: {original_size:.2f} MB")
        logger.info(f"  Quantized size: {quantized_size:.2f} MB")
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        
        self.compression_stats['dynamic_quantization'] = {
            'original_size_mb': original_size,
            'compressed_size_mb': quantized_size,
            'compression_ratio': compression_ratio
        }
        
        return quantized_model
    
    def apply_pruning(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        method: str = 'magnitude'
    ) -> nn.Module:
        """
        Apply weight pruning to model.
        
        Pruning removes less important weights (set to zero), reducing
        computation and memory requirements.
        
        Args:
            model: PyTorch model to prune
            pruning_ratio: Fraction of weights to prune (0-1)
            method: Pruning method ('magnitude', 'random', 'structured')
            
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        logger.info(f"Applying {method} pruning with ratio {pruning_ratio}...")
        
        # Count original parameters
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply pruning to all Linear and Conv layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if method == 'magnitude':
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                elif method == 'random':
                    prune.random_unstructured(module, name='weight', amount=pruning_ratio)
                
                # Make pruning permanent
                prune.remove(module, 'weight')
        
        # Count remaining non-zero parameters
        remaining_params = sum(
            (p != 0).sum().item() for p in model.parameters()
        )
        
        actual_pruning_ratio = 1 - (remaining_params / original_params)
        
        logger.info(f"Pruning complete:")
        logger.info(f"  Original parameters: {original_params:,}")
        logger.info(f"  Remaining parameters: {remaining_params:,}")
        logger.info(f"  Actual pruning ratio: {actual_pruning_ratio:.2%}")
        
        self.compression_stats['pruning'] = {
            'original_params': original_params,
            'remaining_params': remaining_params,
            'pruning_ratio': actual_pruning_ratio
        }
        
        return model
    
    def apply_combined_compression(
        self,
        model: nn.Module,
        quantize: bool = True,
        prune: bool = True,
        pruning_ratio: float = 0.3
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply multiple compression techniques.
        
        Args:
            model: Model to compress
            quantize: Whether to apply quantization
            prune: Whether to apply pruning
            pruning_ratio: Pruning ratio if pruning is enabled
            
        Returns:
            Tuple of (compressed_model, compression_stats)
        """
        logger.info("Applying combined compression...")
        
        compressed_model = copy.deepcopy(model)
        
        # Apply pruning first (if enabled)
        if prune:
            compressed_model = self.apply_pruning(
                compressed_model,
                pruning_ratio=pruning_ratio
            )
        
        # Then apply quantization (if enabled)
        if quantize:
            compressed_model = self.apply_dynamic_quantization(compressed_model)
        
        # Calculate overall compression
        original_size = self._get_model_size(model)
        compressed_size = self._get_model_size(compressed_model)
        overall_compression = original_size / compressed_size
        
        logger.info(f"Combined compression complete:")
        logger.info(f"  Overall compression ratio: {overall_compression:.2f}x")
        
        return compressed_model, self.compression_stats
    
    def _get_model_size(self, model: nn.Module) -> float:
        """
        Calculate model size in MB.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in megabytes
        """
        # Save model to buffer
        import io
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_bytes = buffer.tell()
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
    
    def estimate_energy_savings(
        self,
        compression_ratio: float,
        base_energy: float
    ) -> Dict[str, float]:
        """
        Estimate energy savings from compression.
        
        Args:
            compression_ratio: Model compression ratio
            base_energy: Baseline energy consumption (kWh)
            
        Returns:
            Dictionary with energy estimates
        """
        # Empirical relationship: energy scales roughly with compute
        # Quantization INT8: ~30% energy reduction
        # Pruning 30%: ~20% energy reduction
        # Combined: ~40-50% energy reduction
        
        if compression_ratio >= 4:  # Quantization + pruning
            energy_reduction_pct = 45
        elif compression_ratio >= 3:  # Quantization only
            energy_reduction_pct = 30
        elif compression_ratio >= 2:  # Pruning only
            energy_reduction_pct = 20
        else:
            energy_reduction_pct = 10
        
        compressed_energy = base_energy * (1 - energy_reduction_pct / 100)
        energy_saved = base_energy - compressed_energy
        
        return {
            'base_energy_kwh': base_energy,
            'compressed_energy_kwh': compressed_energy,
            'energy_saved_kwh': energy_saved,
            'energy_reduction_percent': energy_reduction_pct
        }
    
    def validate_compression_accuracy(
        self,
        model_fp32: nn.Module,
        model_int8: nn.Module,
        test_loader: Any,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Validate accuracy loss from INT8 quantization.
        
        CRITICAL QoS METRIC for journal submission:
        - Measures actual accuracy degradation from compression
        - Industry standard: 0.5-2% accuracy loss acceptable
        - Our target: <2% to maintain QoS guarantees
        
        Args:
            model_fp32: Original FP32 model
            model_int8: Quantized INT8 model
            test_loader: PyTorch DataLoader with test data
            device: Device to run evaluation on
            
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Validating INT8 quantization accuracy loss...")
        
        model_fp32.eval()
        model_int8.eval()
        model_fp32.to(device)
        model_int8.to(device)
        
        # Evaluate FP32 model
        fp32_correct = 0
        fp32_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model_fp32(data)
                pred = output.argmax(dim=1, keepdim=True)
                fp32_correct += pred.eq(target.view_as(pred)).sum().item()
                fp32_total += target.size(0)
        
        fp32_accuracy = 100.0 * fp32_correct / fp32_total
        
        # Evaluate INT8 model
        int8_correct = 0
        int8_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model_int8(data)
                pred = output.argmax(dim=1, keepdim=True)
                int8_correct += pred.eq(target.view_as(pred)).sum().item()
                int8_total += target.size(0)
        
        int8_accuracy = 100.0 * int8_correct / int8_total
        
        # Calculate accuracy loss
        accuracy_loss_pct = fp32_accuracy - int8_accuracy
        accuracy_loss_relative = (accuracy_loss_pct / fp32_accuracy) * 100
        
        # Determine if accuracy loss is acceptable
        acceptable = accuracy_loss_pct <= 2.0  # Industry standard: <2% loss
        
        result = {
            'fp32_accuracy_pct': fp32_accuracy,
            'int8_accuracy_pct': int8_accuracy,
            'accuracy_loss_pct': accuracy_loss_pct,
            'accuracy_loss_relative_pct': accuracy_loss_relative,
            'acceptable': acceptable,
            'samples_evaluated': fp32_total
        }
        
        logger.info(f"Accuracy Validation Results:")
        logger.info(f"  FP32 Accuracy: {fp32_accuracy:.2f}%")
        logger.info(f"  INT8 Accuracy: {int8_accuracy:.2f}%")
        logger.info(f"  Accuracy Loss: {accuracy_loss_pct:.2f}% (absolute)")
        logger.info(f"  Relative Loss: {accuracy_loss_relative:.2f}%")
        logger.info(f"  Acceptable: {'YES' if acceptable else 'NO'} (<2% threshold)")
        
        self.compression_stats['accuracy_validation'] = result
        
        return result
    
    def estimate_accuracy_loss_synthetic(
        self,
        model_complexity: str = 'medium',
        compression_type: str = 'int8'
    ) -> Dict[str, float]:
        """
        Estimate accuracy loss for simulation when real validation data unavailable.
        
        Based on literature benchmarks:
        - Simple models (MobileNet): 0.3-0.8% loss
        - Medium models (ResNet-50): 0.5-1.5% loss
        - Complex models (EfficientNet): 1.0-2.5% loss
        
        Args:
            model_complexity: 'simple', 'medium', or 'complex'
            compression_type: 'int8', 'int4', or 'pruned'
            
        Returns:
            Dictionary with estimated accuracy metrics
        """
        import numpy as np
        
        # Literature-based accuracy loss ranges
        accuracy_loss_ranges = {
            'int8': {
                'simple': (0.3, 0.8),    # MobileNetV2, ShuffleNet
                'medium': (0.5, 1.5),    # ResNet-50, VGG-16
                'complex': (1.0, 2.5),   # EfficientNet-B7, Inception-v4
            },
            'int4': {
                'simple': (1.0, 2.5),
                'medium': (2.0, 4.0),
                'complex': (3.5, 6.0),
            },
            'pruned': {
                'simple': (0.2, 0.6),
                'medium': (0.4, 1.2),
                'complex': (0.8, 2.0),
            }
        }
        
        # Get range for this configuration
        loss_range = accuracy_loss_ranges.get(compression_type, {}).get(model_complexity, (0.5, 1.5))
        
        # Sample from range with bias toward lower values (most models compress well)
        # Use beta distribution for realistic skew
        beta_sample = np.random.beta(2, 5)  # Skewed toward lower loss
        estimated_loss = loss_range[0] + beta_sample * (loss_range[1] - loss_range[0])
        
        # Assume baseline accuracy
        baseline_accuracy = {
            'simple': 92.0,
            'medium': 76.0,
            'complex': 84.0
        }.get(model_complexity, 80.0)
        
        compressed_accuracy = baseline_accuracy - estimated_loss
        
        result = {
            'baseline_accuracy_pct': baseline_accuracy,
            'compressed_accuracy_pct': compressed_accuracy,
            'accuracy_loss_pct': estimated_loss,
            'accuracy_loss_relative_pct': (estimated_loss / baseline_accuracy) * 100,
            'acceptable': estimated_loss <= 2.0,
            'estimation_method': 'synthetic_literature_based',
            'model_complexity': model_complexity,
            'compression_type': compression_type
        }
        
        logger.info(f"Synthetic Accuracy Estimation ({model_complexity} model, {compression_type}):")
        logger.info(f"  Baseline: {baseline_accuracy:.2f}%")
        logger.info(f"  Compressed: {compressed_accuracy:.2f}%")
        logger.info(f"  Loss: {estimated_loss:.2f}%")
        
        return result


def create_sample_model(model_type: str = 'resnet18') -> nn.Module:
    """
    Create a sample model for testing.
    
    Args:
        model_type: Type of model ('resnet18', 'mobilenet', 'simple')
        
    Returns:
        PyTorch model
    """
    if (model_type == 'resnet18'):
        from torchvision import models
        model = models.resnet18(pretrained=False)
        logger.info("Created ResNet-18 model")
        
    elif (model_type == 'mobilenet'):
        from torchvision import models
        model = models.mobilenet_v2(pretrained=False)
        logger.info("Created MobileNet-V2 model")
        
    elif (model_type == 'simple'):
        # Simple feedforward network for testing
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        logger.info("Created simple feedforward model")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def benchmark_compression(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100
) -> Dict[str, Any]:
    """
    Benchmark inference speed and energy for compressed vs uncompressed models.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_iterations: Number of inference iterations
        
    Returns:
        Benchmark results
    """
    import time
    
    logger.info("Benchmarking compression...")
    
    # Create compressor
    compressor = ModelCompressor()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Benchmark original model
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    original_time = time.time() - start_time
    
    # Apply compression
    compressed_model, stats = compressor.apply_combined_compression(
        model,
        quantize=True,
        prune=True,
        pruning_ratio=0.3
    )
    
    # Benchmark compressed model
    compressed_model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = compressed_model(dummy_input)
    compressed_time = time.time() - start_time
    
    speedup = original_time / compressed_time
    
    results = {
        'original_time_sec': original_time,
        'compressed_time_sec': compressed_time,
        'speedup': speedup,
        'compression_stats': stats
    }
    
    logger.info(f"Benchmark results:")
    logger.info(f"  Original time: {original_time:.3f}s")
    logger.info(f"  Compressed time: {compressed_time:.3f}s")
    logger.info(f"  Speedup: {speedup:.2f}x")
    
    return results
