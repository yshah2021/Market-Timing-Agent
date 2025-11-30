"""
Configuration management for the Market Timing Agents system
"""

from typing import Dict, Any
import json
from pathlib import Path

class Config:
    """Configuration manager"""
    
    DEFAULT_CONFIG = {
        "quality_screening": {
            "max_candidates": 25,
            "min_quality_score": 7.0,
            "screener_url": "https://www.screener.in/screens/3327459/star-stocks/"
        },
        "entry_timing": {
            "min_confidence": 50.0,
            "min_risk_reward": 1.0,
            "max_pe_ratio_multiplier": 2.5,
            "rsi_oversold": 30,
            "rsi_overbought": 70
        },
        "exit_management": {
            "default_stop_loss_pct": 6.0,
            "default_target_pct": 12.0,
            "min_risk_reward": 1.0
        },
        "database": {
            "path": "market_timing_agents.db"
        },
        "logging": {
            "level": "INFO",
            "dir": "logs"
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config file
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file"""
        config_to_save = config or self.config
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value