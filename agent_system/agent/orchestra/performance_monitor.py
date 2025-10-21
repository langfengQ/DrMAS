"""
Performance monitoring utilities for multi-agent orchestra.

Usage:
    from agent_system.agent.orchestra.performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor(enabled=True)
    
    with monitor.measure("agent_name"):
        # Your code here
        pass
    
    stats = monitor.get_stats()
"""

import time
from typing import Dict, Optional, Any
from contextlib import contextmanager
import numpy as np


class PerformanceMonitor:
    """Lightweight performance monitoring for agent orchestra."""
    
    def __init__(self, enabled: bool = False):
        """
        Initialize performance monitor.
        
        Args:
            enabled: Whether to enable performance monitoring
        """
        self.enabled = enabled
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.step_count = 0
        
    def reset(self):
        """Reset all statistics."""
        self.stats.clear()
        self.step_count = 0
    
    @contextmanager
    def measure(self, name: str, metadata: Optional[Dict] = None):
        """
        Context manager to measure execution time.
        
        Args:
            name: Name of the operation/agent being measured
            metadata: Optional metadata to track (e.g., batch_size, active_samples)
        
        Example:
            with monitor.measure("Search Agent", {"batch_size": 64, "active": 32}):
                # Agent execution code
                pass
        """
        if not self.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self._record(name, elapsed, metadata)
    
    def _record(self, name: str, elapsed: float, metadata: Optional[Dict] = None):
        """Record a timing measurement."""
        if name not in self.stats:
            self.stats[name] = {
                'count': 0,
                'total_time': 0.0,
                'times': [],
                'metadata': []
            }
        
        self.stats[name]['count'] += 1
        self.stats[name]['total_time'] += elapsed
        self.stats[name]['times'].append(elapsed)
        
        if metadata:
            self.stats[name]['metadata'].append(metadata)
    
    def record_step(self):
        """Record completion of a step."""
        self.step_count += 1
    
    def get_stats(self, reset: bool = False) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            reset: Whether to reset stats after retrieval
            
        Returns:
            Dictionary containing performance statistics
        """
        result = {}
        
        for name, data in self.stats.items():
            times = np.array(data['times'])
            result[name] = {
                'count': data['count'],
                'total_time': data['total_time'],
                'mean_time': np.mean(times) if len(times) > 0 else 0.0,
                'median_time': np.median(times) if len(times) > 0 else 0.0,
                'std_time': np.std(times) if len(times) > 0 else 0.0,
                'min_time': np.min(times) if len(times) > 0 else 0.0,
                'max_time': np.max(times) if len(times) > 0 else 0.0,
            }
            
            # Compute average metadata if available
            if data['metadata']:
                metadata_keys = data['metadata'][0].keys()
                avg_metadata = {}
                for key in metadata_keys:
                    values = [m[key] for m in data['metadata'] if key in m]
                    if values and isinstance(values[0], (int, float)):
                        avg_metadata[f'avg_{key}'] = np.mean(values)
                result[name]['metadata'] = avg_metadata
        
        result['step_count'] = self.step_count
        
        if reset:
            self.reset()
        
        return result
    
    def print_stats(self, reset: bool = False):
        """
        Print formatted performance statistics.
        
        Args:
            reset: Whether to reset stats after printing
        """
        if not self.enabled:
            print("Performance monitoring is disabled.")
            return
        
        stats = self.get_stats(reset=reset)
        
        print("\n" + "="*80)
        print(f"Performance Statistics (Steps: {stats['step_count']})")
        print("="*80)
        
        # Sort by total time (descending)
        sorted_items = sorted(
            [(k, v) for k, v in stats.items() if k != 'step_count'],
            key=lambda x: x[1].get('total_time', 0),
            reverse=True
        )
        
        print(f"{'Operation':<30} {'Count':>8} {'Total(s)':>10} {'Mean(s)':>10} {'Std(s)':>10}")
        print("-"*80)
        
        for name, data in sorted_items:
            print(f"{name:<30} {data['count']:>8} {data['total_time']:>10.3f} "
                  f"{data['mean_time']:>10.3f} {data['std_time']:>10.3f}")
            
            # Print metadata if available
            if 'metadata' in data and data['metadata']:
                metadata_str = ", ".join([f"{k}={v:.1f}" for k, v in data['metadata'].items()])
                print(f"  └─ {metadata_str}")
        
        print("="*80 + "\n")
    
    def get_throughput(self, batch_size: int) -> float:
        """
        Calculate overall throughput.
        
        Args:
            batch_size: Batch size used
            
        Returns:
            Throughput in samples/second
        """
        if not self.enabled or self.step_count == 0:
            return 0.0
        
        total_time = sum(data['total_time'] for data in self.stats.values())
        total_samples = self.step_count * batch_size
        
        return total_samples / total_time if total_time > 0 else 0.0


class TimingContext:
    """Simple timing context for inline measurements."""
    
    def __init__(self, name: str, enabled: bool = True, verbose: bool = False):
        self.name = name
        self.enabled = enabled
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        if self.enabled:
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            self.elapsed = time.perf_counter() - self.start_time
            if self.verbose:
                print(f"{self.name}: {self.elapsed:.3f}s")


# Singleton instance for easy access
_global_monitor = PerformanceMonitor(enabled=False)


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


def enable_global_monitor():
    """Enable global performance monitoring."""
    _global_monitor.enabled = True


def disable_global_monitor():
    """Disable global performance monitoring."""
    _global_monitor.enabled = False
