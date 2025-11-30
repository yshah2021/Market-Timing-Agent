"""
Market Timing Agents - Production Trading System

A comprehensive multi-agent trading system with production-grade features for Indian stock markets.

Project Structure:
- agents/    : Core trading agents (screening, timing, exit management)
- utils/     : Production utilities (logging, safety, backtesting)  
- tests/     : Comprehensive testing framework
"""

# Import main components for easy access
from agents import (
    Agent1ScreeningEngine,
    Agent2EntryTimingEngine, 
    Agent3ExitManager,
    run_screening,
    run_timing_analysis,
    run_exit_planning
)

from utils import (
    SystemLogger,
    SafetyGuardrails,
    KPICalculator,
    EnhancedBacktestEngine,
    WorkingBacktester
)

from tests import (
    run_all_tests,
    MarketTimingSystemIntegrationTest
)

# Main system
from market_timing_system import MarketTimingSystem

__version__ = "2.0.0"
__author__ = "Market Timing Agents Team"

__all__ = [
    # Core Agents
    'Agent1ScreeningEngine',
    'Agent2EntryTimingEngine', 
    'Agent3ExitManager',
    
    # Agent Functions
    'run_screening',
    'run_timing_analysis', 
    'run_exit_planning',
    
    # Main System
    'MarketTimingSystem',
    
    # Utilities
    'EnterpriseLogger',
    'SafetyGuardrails',
    'KPICalculator',
    'EnhancedBacktestEngine',
    'WorkingBacktester',
    
    # Testing
    'run_all_tests',
    'MarketTimingSystemIntegrationTest'
]