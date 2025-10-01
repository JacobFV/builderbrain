"""
Data loading utilities for BuilderBrain Dashboard.

Handles loading training history, metrics, and configuration data from BuilderBrain.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import psutil

# Add parent directory to path for BuilderBrain imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class DataLoader:
    """Data loader for BuilderBrain training and runtime data."""

    def __init__(self):
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        self.training_history_file = os.path.join(self.base_path, 'training_history.json')

    def load_training_history(self) -> Optional[Dict[str, Any]]:
        """Load training history from JSON file."""
        try:
            if os.path.exists(self.training_history_file):
                with open(self.training_history_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading training history: {e}")
            return None

    def get_training_metrics_df(self) -> pd.DataFrame:
        """Convert training history to pandas DataFrame."""
        history = self.load_training_history()
        if not history:
            return pd.DataFrame()

        df_data = []

        for i, (total_loss, task_loss, constraint_losses, dual_vars) in enumerate(zip(
            history.get('total_loss', []),
            history.get('task_loss', []),
            history.get('constraint_losses', {}).values(),
            history.get('dual_variables', [])
        )):
            df_data.append({
                'step': i,
                'total_loss': total_loss,
                'task_loss': task_loss,
                'epoch': i // 10,  # Assuming 10 steps per epoch
                **{f"constraint_{k}": v for k, v in constraint_losses.items()},
                **{f"dual_{k}": v for k, v in dual_vars.items()},
            })

        return pd.DataFrame(df_data)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'active_processes': len(psutil.pids()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error getting system metrics: {e}")
            return {}

    def get_model_config(self) -> Optional[Dict[str, Any]]:
        """Load model configuration."""
        config_file = os.path.join(self.base_path, 'configs', 'small.yaml')
        if not os.path.exists(config_file):
            return None

        try:
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return None

    def get_constraint_config(self) -> Optional[Dict[str, Any]]:
        """Load constraint configuration."""
        config = self.get_model_config()
        if config:
            return config.get('constraints', {})
        return None

    def get_training_config(self) -> Optional[Dict[str, Any]]:
        """Load training configuration."""
        config = self.get_model_config()
        if config:
            return config.get('training', {})
        return None

    def get_latest_training_metrics(self) -> Dict[str, Any]:
        """Get the most recent training metrics."""
        df = self.get_training_metrics_df()
        if df.empty:
            return {}

        latest = df.iloc[-1]

        return {
            'step': int(latest['step']),
            'total_loss': float(latest['total_loss']),
            'task_loss': float(latest['task_loss']),
            'epoch': int(latest['epoch']),
            'dual_variables': {
                col.replace('dual_', ''): float(latest[col])
                for col in df.columns if col.startswith('dual_')
            },
            'constraint_losses': {
                col.replace('constraint_', ''): float(latest[col])
                for col in df.columns if col.startswith('constraint_')
            }
        }

    def get_constraint_violation_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations."""
        df = self.get_training_metrics_df()
        if df.empty:
            return {}

        summary = {}

        # Calculate violation rates for each constraint
        for col in df.columns:
            if col.startswith('constraint_'):
                constraint_name = col.replace('constraint_', '')
                values = df[col].dropna()

                if len(values) > 0:
                    # Calculate how often constraint target is violated
                    # Assuming target is 0 for most constraints
                    violations = (values > 0.1).sum()  # Threshold for violation
                    violation_rate = violations / len(values)

                    summary[constraint_name] = {
                        'current_value': float(values.iloc[-1]),
                        'violation_rate': float(violation_rate),
                        'trend': 'improving' if values.iloc[-10:].mean() < values.iloc[:10].mean() else 'degrading'
                    }

        return summary

    def get_training_progress_summary(self) -> Dict[str, Any]:
        """Get overall training progress summary."""
        history = self.load_training_history()
        if not history:
            return {}

        total_loss = history.get('total_loss', [])
        if not total_loss:
            return {}

        # Calculate improvement metrics
        recent_loss = total_loss[-10:] if len(total_loss) >= 10 else total_loss
        early_loss = total_loss[:10] if len(total_loss) >= 10 else total_loss

        avg_recent = np.mean(recent_loss)
        avg_early = np.mean(early_loss)
        improvement = (avg_early - avg_recent) / avg_early if avg_early > 0 else 0

        return {
            'total_steps': len(total_loss),
            'current_loss': float(total_loss[-1]),
            'best_loss': float(min(total_loss)),
            'improvement_rate': float(improvement),
            'is_converging': improvement > 0.01,  # 1% improvement threshold
            'epochs_completed': len(total_loss) // 10  # Assuming 10 steps per epoch
        }
