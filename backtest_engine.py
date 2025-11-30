#!/usr/bin/env python3
"""
Comprehensive Backtesting Engine for Market Timing Agents
Supports historical analysis of blue-chip stocks with realistic trading simulation
"""

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import json
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import get_logger
from agents.quality_screening_agent import QualityScreeningAgent
from agents.entry_timing_agent import EntryTimingAgent
from agents.exit_management_agent import ExitManagementAgent
from agents.entry_timing_agent import EntryTimingAgent
from agents.exit_management_agent import ExitManagementAgent
from data_cache import DataCache

@dataclass
class Position:
    """Represents a trading position"""
    ticker: str
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    return_pct: Optional[float] = None

@dataclass
class BacktestResults:
    """Contains comprehensive backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    avg_holding_period: float
    positions: List[Position]
    monthly_returns: Dict
    trade_analysis: Dict

class ComprehensiveBacktester:
    """
    Advanced backtesting engine with realistic trading simulation
    """
    
    def __init__(self, initial_capital: float = 1000000, max_position_size: float = 0.15):
        self.logger = get_logger("Backtester")
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        
        # Initialize agents
        self.quality_agent = QualityScreeningAgent()
        self.entry_agent = EntryTimingAgent()
        self.exit_agent = ExitManagementAgent()
        
        # Initialize data cache
        self.data_cache = DataCache()
        
        # Track positions and performance
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity_curve: List[Dict] = []
        
        self.logger.info(f"Backtester initialized with ₹{initial_capital:,.0f} capital")
    
    def run_backtest(self, 
                    tickers: List[str], 
                    start_date: str = "2023-01-01", 
                    end_date: str = "2024-12-31") -> BacktestResults:
        """
        Run comprehensive backtest on given tickers for specified period
        
        Args:
            tickers: List of ticker symbols to backtest
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            BacktestResults with comprehensive analysis
        """
        
        self.logger.info(f"Starting backtest: {tickers} from {start_date} to {end_date}")
        
        # Fetch historical data for all tickers
        historical_data = self._fetch_historical_data(tickers, start_date, end_date)
        if not historical_data:
            self.logger.error("No historical data available")
            return self._generate_empty_results()
        
        # Apply quality screening to filter the best candidates
        qualified_tickers = self._apply_quality_screening(list(historical_data.keys()), historical_data)
        if not qualified_tickers:
            self.logger.warning("No tickers passed quality screening")
            qualified_tickers = list(historical_data.keys())  # Fallback to all tickers
        
        self.logger.info(f"Quality screening: {len(tickers)} → {len(qualified_tickers)} candidates")
        
        # Create trading calendar (business days)
        trading_days = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Simulate day-by-day trading
        for current_date in trading_days:
            # Convert to timestamp for timezone-aware comparison
            current_date = pd.Timestamp(current_date).tz_localize(None)
            try:
                # Check for exit signals on existing positions
                self._process_exits(current_date, historical_data)
                
                # Check for new entry signals (only on qualified tickers)
                self._process_entries(current_date, historical_data, qualified_tickers)
                
                # Update equity curve
                self._update_equity_curve(current_date, historical_data)
                
                # Log progress periodically
                if current_date.day == 1:  # First day of each month
                    self.logger.info(f"Progress: {current_date.strftime('%Y-%m')}, "
                                   f"Open positions: {len(self.open_positions)}, "
                                   f"Capital: ₹{self.current_capital:,.0f}")
                
            except Exception as e:
                self.logger.error(f"Error processing {current_date}: {e}")
                continue
        
        # Close any remaining positions at end date
        self._close_remaining_positions(trading_days[-1], historical_data)
        
        # Generate comprehensive results
        results = self._calculate_backtest_results()
        
        self.logger.info(f"Backtest completed: {results.total_trades} trades, "
                        f"{results.win_rate:.1f}% win rate, "
                        f"{results.total_return:.1f}% total return")
        
        return results
    
    def _fetch_historical_data(self, 
                              tickers: List[str], 
                              start_date: str, 
                              end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch historical data using cache system"""
        
        data = {}
        
        self.logger.info(f"Fetching historical data for {len(tickers)} tickers using cache")
        
        for ticker in tickers:
            try:
                # Use data cache instead of direct yfinance calls
                hist = self.data_cache.get_historical_data(ticker, start_date, end_date)
                
                if hist is not None and not hist.empty:
                    data[ticker] = hist
                    self.logger.info(f"Loaded {len(hist)} days of data for {ticker}")
                else:
                    self.logger.warning(f"No data found for {ticker}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {e}")
        
        # Show cache statistics
        cache_info = self.data_cache.get_cache_info()
        self.logger.info(f"Cache status: {cache_info['total_files']} files, "
                        f"{cache_info['cache_size_mb']} MB total")
        
        return data
    
    def _apply_quality_screening(self, tickers: List[str], historical_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Apply quality screening to filter the best ticker candidates"""
        
        qualified_tickers = []
        
        for ticker in tickers:
            if ticker not in historical_data:
                continue
                
            try:
                # Get latest data for quality analysis
                df = historical_data[ticker]
                latest_data = df.iloc[-50:] if len(df) >= 50 else df  # Last 50 days
                
                if latest_data.empty:
                    continue
                
                # Prepare data for quality agent
                stock_data = {
                    'ticker': ticker,
                    'current_price': latest_data['Close'].iloc[-1],
                    'price_change_1d': ((latest_data['Close'].iloc[-1] / latest_data['Close'].iloc[-2]) - 1) * 100 if len(latest_data) > 1 else 0,
                    'price_change_5d': ((latest_data['Close'].iloc[-1] / latest_data['Close'].iloc[-6]) - 1) * 100 if len(latest_data) > 5 else 0,
                    'price_change_30d': ((latest_data['Close'].iloc[-1] / latest_data['Close'].iloc[-31]) - 1) * 100 if len(latest_data) > 30 else 0,
                    'volume': latest_data['Volume'].iloc[-1] if 'Volume' in latest_data else 1000000,
                    'rsi': latest_data['RSI'].iloc[-1] if 'RSI' in latest_data else 50,
                    'sma_50': latest_data['SMA_50'].iloc[-1] if 'SMA_50' in latest_data else latest_data['Close'].iloc[-1],
                    'sma_200': latest_data['SMA_200'].iloc[-1] if 'SMA_200' in latest_data else latest_data['Close'].iloc[-1]
                }
                
                # Use QualityScreeningAgent to evaluate
                quality_result = self.quality_agent.screen_stock_quality(
                    stock_data=stock_data,
                    historical_data=latest_data
                )
                
                if quality_result.get('action') == 'APPROVE':
                    qualified_tickers.append(ticker)
                    self.logger.info(f"Quality screening: {ticker} APPROVED (Score: {quality_result.get('score', 0):.1f})")
                else:
                    self.logger.info(f"Quality screening: {ticker} REJECTED (Score: {quality_result.get('score', 0):.1f})")
                    
            except Exception as e:
                self.logger.error(f"Error in quality screening for {ticker}: {e}")
                # Include in qualified list as fallback
                qualified_tickers.append(ticker)
        
        return qualified_tickers
    
    def _process_entries(self, 
                        current_date: datetime, 
                        historical_data: Dict[str, pd.DataFrame], 
                        tickers: List[str]):
        """Process potential entry signals for current date"""
        
        # Don't enter new positions if we're at max capacity
        if len(self.open_positions) >= 3:  # Max 3 concurrent positions
            return
        
        for ticker in tickers:
            if ticker not in historical_data:
                continue
                
            # Skip if already have position in this ticker
            if any(pos.ticker == ticker for pos in self.open_positions):
                continue
            
            df = historical_data[ticker]
            
            # Get data up to current date
            # Ensure timezone compatibility
            df_index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            current_data = df[df_index <= current_date]
            if len(current_data) < 50:  # Need minimum data for indicators
                continue
            
            # Get current price and indicators
            current_price = current_data['Close'].iloc[-1]
            
            # Apply entry logic using actual agents
            entry_signal = self._evaluate_entry_signal(ticker, current_data)
            
            if entry_signal and entry_signal.get('should_buy', False):
                self._create_position(ticker, current_date, current_price, entry_signal)
    
    def _evaluate_entry_signal(self, 
                              ticker: str,
                              current_data: pd.DataFrame) -> Dict:
        """Evaluate whether to enter a position using actual agents"""
        
        try:
            # Prepare market data for agent
            market_data = {
                'current_price': current_data['Close'].iloc[-1],
                'volume': current_data['Volume'].iloc[-1] if 'Volume' in current_data else 0,
                'rsi': current_data['RSI'].iloc[-1],
                'macd': current_data['MACD'].iloc[-1],
                'macd_signal': current_data['MACD_Signal'].iloc[-1],
                'sma_50': current_data['SMA_50'].iloc[-1],
                'sma_200': current_data['SMA_200'].iloc[-1],
                'price_change': ((current_data['Close'].iloc[-1] / current_data['Close'].iloc[-2]) - 1) * 100 if len(current_data) > 1 else 0
            }
            
            # Use EntryTimingAgent to evaluate entry signal
            candidate_data = {
                'ticker': ticker,
                'company_name': ticker,  # For agent processing
                'current_price': current_data['Close'].iloc[-1],
                'volume': current_data['Volume'].iloc[-1] if 'Volume' in current_data.columns else 0
            }
            
            # Get entry signals from agent (expects list input)
            entry_decisions = self.entry_agent.analyze_entry_signals([candidate_data])
            
            if entry_decisions and len(entry_decisions) > 0:
                entry_decision = entry_decisions[0]
            else:
                return None
            
            # Check if the signal is a BUY signal (BUY_NOW or BUY_SOON)
            signal_action = entry_decision.get('entry_signal', '')
            is_buy_signal = signal_action in ['BUY_NOW', 'BUY_SOON']
            
            return {
                'should_buy': is_buy_signal,
                'confidence': entry_decision.get('entry_confidence', 0),
                'score': entry_decision.get('entry_confidence', 0),
                'reasons': [entry_decision.get('reasoning', f'Agent analysis: {signal_action}')],
                'agent_decision': entry_decision
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating entry signal for {ticker}: {e}")
            return {
                'should_buy': False,
                'confidence': 0,
                'score': 0,
                'reasons': [f"Error: {str(e)}"],
                'agent_decision': None
            }
    
    def _create_position(self, 
                        ticker: str, 
                        entry_date: datetime, 
                        entry_price: float, 
                        signal: Dict):
        """Create a new trading position"""
        
        # Calculate position size (max 15% of capital per position)
        position_value = min(self.current_capital * self.max_position_size, 
                           self.current_capital / 3)  # Spread across max 3 positions
        quantity = int(position_value / entry_price)
        
        if quantity == 0:
            return
        
        # Set stop loss and targets
        stop_loss = entry_price * 0.92  # 8% stop loss
        target_1 = entry_price * 1.08   # 8% target
        target_2 = entry_price * 1.15   # 15% target  
        target_3 = entry_price * 1.25   # 25% target
        
        position = Position(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3
        )
        
        self.open_positions.append(position)
        self.current_capital -= (quantity * entry_price)
        
        self.logger.info(f"ENTRY: {ticker} @ ₹{entry_price:.2f} x {quantity} shares "
                        f"(Confidence: {signal['confidence']:.1f}%)")
    
    def _process_exits(self, 
                      current_date: datetime, 
                      historical_data: Dict[str, pd.DataFrame]):
        """Process exit signals for existing positions"""
        
        positions_to_close = []
        
        for position in self.open_positions:
            if position.ticker not in historical_data:
                continue
            
            df = historical_data[position.ticker]
            # Ensure timezone compatibility
            df_index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            current_data = df[df_index <= current_date]
            
            if current_data.empty:
                continue
                
            current_price = current_data['Close'].iloc[-1]
            holding_days = (current_date - position.entry_date).days
            
            # Use ExitManagementAgent to determine exit decision
            try:
                market_data = {
                    'current_price': current_price,
                    'volume': current_data['Volume'].iloc[-1] if 'Volume' in current_data else 0,
                    'rsi': current_data['RSI'].iloc[-1],
                    'macd': current_data['MACD'].iloc[-1],
                    'macd_signal': current_data['MACD_Signal'].iloc[-1],
                    'sma_50': current_data['SMA_50'].iloc[-1],
                    'sma_200': current_data['SMA_200'].iloc[-1],
                    'entry_price': position.entry_price,
                    'holding_days': holding_days,
                    'stop_loss': position.stop_loss,
                    'target_1': position.target_1,
                    'target_2': position.target_2,
                    'target_3': position.target_3
                }
                
                exit_decision = self.exit_agent.check_exit_conditions(
                    position={
                        'ticker': position.ticker,
                        'entry_price': position.entry_price,
                        'entry_date': position.entry_date.strftime('%Y-%m-%d'),
                        'quantity': position.quantity,
                        'stop_loss': position.stop_loss,
                        'target_1': position.target_1,
                        'target_2': position.target_2,
                        'target_3': position.target_3
                    },
                    current_price=current_price
                )
                
                if exit_decision.get('should_exit', False):
                    exit_reason = exit_decision.get('exit_reason', 'AGENT_EXIT')
                    self._close_position(position, current_date, current_price, exit_reason)
                    positions_to_close.append(position)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating exit for {position.ticker}: {e}")
                # Fallback to basic stop loss
                if current_price <= position.stop_loss:
                    self._close_position(position, current_date, current_price, "STOP_LOSS_FALLBACK")
                    positions_to_close.append(position)
        
        # Remove closed positions
        for position in positions_to_close:
            self.open_positions.remove(position)
    
    def _close_position(self, 
                       position: Position, 
                       exit_date: datetime, 
                       exit_price: float, 
                       exit_reason: str):
        """Close a trading position"""
        
        # Update position with exit data
        position.exit_date = exit_date
        position.exit_price = exit_price
        position.exit_reason = exit_reason
        
        # Calculate P&L
        gross_pnl = (exit_price - position.entry_price) * position.quantity
        brokerage = max(20, 0.0003 * position.quantity * (position.entry_price + exit_price))  # 0.03% brokerage
        position.pnl = gross_pnl - brokerage
        position.return_pct = (position.pnl / (position.entry_price * position.quantity)) * 100
        
        # Update capital
        proceeds = (position.quantity * exit_price) - brokerage/2
        self.current_capital += proceeds
        
        # Add to closed positions
        self.closed_positions.append(position)
        
        holding_days = (exit_date - position.entry_date).days
        self.logger.info(f"EXIT: {position.ticker} @ ₹{exit_price:.2f} "
                        f"({exit_reason}) | P&L: ₹{position.pnl:,.0f} "
                        f"({position.return_pct:+.1f}%) | {holding_days} days")
    
    def _close_remaining_positions(self, 
                                  end_date: datetime, 
                                  historical_data: Dict[str, pd.DataFrame]):
        """Close any remaining open positions at end of backtest"""
        
        for position in self.open_positions[:]:  # Copy list to avoid modification during iteration
            if position.ticker in historical_data:
                df = historical_data[position.ticker]
                # Ensure timezone compatibility  
                df_index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                final_data = df[df_index <= end_date]
                if not final_data.empty:
                    final_price = final_data['Close'].iloc[-1]
                    self._close_position(position, end_date, final_price, "END_BACKTEST")
        
        self.open_positions.clear()
    
    def _update_equity_curve(self, 
                           current_date: datetime, 
                           historical_data: Dict[str, pd.DataFrame]):
        """Update daily equity curve"""
        
        # Calculate current portfolio value
        position_value = 0
        for position in self.open_positions:
            if position.ticker in historical_data:
                df = historical_data[position.ticker]
                # Ensure timezone compatibility
                df_index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                current_data = df[df_index <= current_date]
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    position_value += position.quantity * current_price
        
        total_equity = self.current_capital + position_value
        
        self.equity_curve.append({
            'date': current_date,
            'equity': total_equity,
            'cash': self.current_capital,
            'positions_value': position_value,
            'open_positions': len(self.open_positions)
        })
    
    def _calculate_backtest_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        if not self.closed_positions:
            return self._generate_empty_results()
        
        # Basic statistics
        total_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100
        
        # Returns
        final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Average holding period
        holding_periods = [(p.exit_date - p.entry_date).days for p in self.closed_positions]
        avg_holding_period = np.mean(holding_periods)
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns()
        
        # Trade analysis
        trade_analysis = self._analyze_trades()
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            avg_holding_period=avg_holding_period,
            positions=self.closed_positions,
            monthly_returns=monthly_returns,
            trade_analysis=trade_analysis
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.equity_curve:
            return 0.0
        
        equity_values = [point['equity'] for point in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Daily returns
        daily_returns = []
        for i in range(1, len(self.equity_curve)):
            prev_equity = self.equity_curve[i-1]['equity']
            curr_equity = self.equity_curve[i]['equity']
            daily_return = (curr_equity - prev_equity) / prev_equity
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0.0
        
        avg_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        
        if std_daily_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming 252 trading days)
        return (avg_daily_return * 252) / (std_daily_return * np.sqrt(252))
    
    def _calculate_monthly_returns(self) -> Dict:
        """Calculate monthly returns"""
        if not self.equity_curve:
            return {}
        
        monthly_data = {}
        
        # Group equity data by month
        for point in self.equity_curve:
            month_key = point['date'].strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(point['equity'])
        
        monthly_returns = {}
        prev_month_end = self.initial_capital
        
        for month, equities in sorted(monthly_data.items()):
            month_end = equities[-1]
            monthly_return = ((month_end - prev_month_end) / prev_month_end) * 100
            monthly_returns[month] = monthly_return
            prev_month_end = month_end
        
        return monthly_returns
    
    def _analyze_trades(self) -> Dict:
        """Analyze trade performance"""
        if not self.closed_positions:
            return {}
        
        profits = [p.pnl for p in self.closed_positions if p.pnl > 0]
        losses = [p.pnl for p in self.closed_positions if p.pnl < 0]
        
        analysis = {
            'avg_profit': np.mean(profits) if profits else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'max_profit': max([p.pnl for p in self.closed_positions]),
            'max_loss': min([p.pnl for p in self.closed_positions]),
            'profit_factor': sum(profits) / abs(sum(losses)) if losses else float('inf'),
            'avg_return_pct': np.mean([p.return_pct for p in self.closed_positions]),
            'best_return_pct': max([p.return_pct for p in self.closed_positions]),
            'worst_return_pct': min([p.return_pct for p in self.closed_positions])
        }
        
        # Exit reason breakdown
        exit_reasons = {}
        for position in self.closed_positions:
            reason = position.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = 0
            exit_reasons[reason] += 1
        
        analysis['exit_reasons'] = exit_reasons
        
        return analysis
    
    def _generate_empty_results(self) -> BacktestResults:
        """Generate empty results for failed backtest"""
        return BacktestResults(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            avg_holding_period=0.0,
            positions=[],
            monthly_returns={},
            trade_analysis={}
        )
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # Initial averages
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Exponential moving averages
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd, signal)
        
        return macd, signal_line
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average"""
        ema = np.zeros_like(prices)
        multiplier = 2 / (period + 1)
        
        # Start with simple moving average
        ema[period-1] = np.mean(prices[:period])
        
        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def print_results(self, results: BacktestResults):
        """Print comprehensive backtest results"""
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE BACKTEST RESULTS")
        print("="*80)
        
        print(f"\n💰 Portfolio Performance:")
        print(f"   Initial Capital: ₹{self.initial_capital:,.0f}")
        final_equity = self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_capital
        print(f"   Final Equity:    ₹{final_equity:,.0f}")
        print(f"   Total Return:    {results.total_return:+.2f}%")
        print(f"   Max Drawdown:    {results.max_drawdown:.2f}%")
        print(f"   Sharpe Ratio:    {results.sharpe_ratio:.2f}")
        
        print(f"\n📈 Trading Statistics:")
        print(f"   Total Trades:    {results.total_trades}")
        print(f"   Winning Trades:  {results.winning_trades}")
        print(f"   Losing Trades:   {results.losing_trades}")
        print(f"   Win Rate:        {results.win_rate:.1f}%")
        print(f"   Avg Holding:     {results.avg_holding_period:.1f} days")
        
        if results.trade_analysis:
            analysis = results.trade_analysis
            print(f"\n💹 Trade Analysis:")
            print(f"   Avg Profit:      ₹{analysis['avg_profit']:,.0f}")
            print(f"   Avg Loss:        ₹{analysis['avg_loss']:,.0f}")
            print(f"   Max Profit:      ₹{analysis['max_profit']:,.0f}")
            print(f"   Max Loss:        ₹{analysis['max_loss']:,.0f}")
            print(f"   Profit Factor:   {analysis['profit_factor']:.2f}")
            
            print(f"\n📊 Return Analysis:")
            print(f"   Avg Return:      {analysis['avg_return_pct']:+.2f}%")
            print(f"   Best Return:     {analysis['best_return_pct']:+.2f}%")
            print(f"   Worst Return:    {analysis['worst_return_pct']:+.2f}%")
            
            if 'exit_reasons' in analysis:
                print(f"\n🚪 Exit Reasons:")
                for reason, count in analysis['exit_reasons'].items():
                    pct = (count / results.total_trades) * 100
                    print(f"   {reason:15s}: {count:3d} ({pct:.1f}%)")
        
        # Show recent trades
        if results.positions:
            print(f"\n📋 Recent Trades (Last 10):")
            recent_trades = sorted(results.positions, key=lambda x: x.exit_date)[-10:]
            for trade in recent_trades:
                print(f"   {trade.ticker:10s} | {trade.exit_date.strftime('%Y-%m-%d')} | "
                      f"₹{trade.pnl:6,.0f} | {trade.return_pct:+6.1f}% | {trade.exit_reason}")
        
        print("\n" + "="*80)

def main():
    """Main function to run backtest on blue-chip stocks"""
    
    # Blue-chip Indian stocks
    blue_chip_stocks = [
        "RELIANCE",    # Reliance Industries
        "TCS",         # Tata Consultancy Services
        "HDFCBANK",    # HDFC Bank
        "INFY",        # Infosys
        "ITC"          # ITC Limited
    ]
    
    print("🚀 Starting Agent-Based Backtest for Blue-Chip Stocks")
    print(f"🤖 Using EntryTimingAgent and ExitManagementAgent")
    print(f"🗄️  Using Data Cache for faster execution")
    print(f"📅 Period: 2023-01-01 to 2024-12-31")
    print(f"📊 Stocks: {', '.join(blue_chip_stocks)}")
    print(f"💰 Capital: ₹10,00,000")
    print()
    
    # Initialize backtester
    backtester = ComprehensiveBacktester(
        initial_capital=1000000,  # ₹10 lakh
        max_position_size=0.15    # Max 15% per position
    )
    
    # Run backtest
    results = backtester.run_backtest(
        tickers=blue_chip_stocks,
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
    
    # Print results
    backtester.print_results(results)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"backtest_results_{timestamp}.json"
    
    results_data = {
        'backtest_config': {
            'tickers': blue_chip_stocks,
            'start_date': "2023-01-01",
            'end_date': "2024-12-31",
            'initial_capital': 1000000,
            'max_position_size': 0.15
        },
        'results': {
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'total_return': results.total_return,
            'max_drawdown': results.max_drawdown,
            'sharpe_ratio': results.sharpe_ratio,
            'trade_analysis': results.trade_analysis,
            'monthly_returns': results.monthly_returns
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_file}")

if __name__ == "__main__":
    main()