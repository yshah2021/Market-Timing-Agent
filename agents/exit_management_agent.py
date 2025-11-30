# Official ADK imports
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
import sqlite3
import yfinance as yf
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from typing import Dict, List, Tuple, Optional, AsyncGenerator
import json

# Import enhanced logging
from utils.logging_config import get_logger

class ExitManagementAgent(BaseAgent):
    """ Exit Planning & Risk Management with Advanced Features"""
    
    model_config = {"arbitrary_types_allowed": True}
    name: str = "ExitManagementAgent"
    logger: Any = None
    db_path: str = "market_timing_agents.db"
    monitoring_count: int = 0
    exit_count: int = 0
    
    def __init__(self, db_path: str = "market_timing_agents.db"):
        super().__init__(name="ExitManagementAgent")
        self.logger = get_logger('agents.exit_management_agent')
        self.db_path = db_path
        self.monitoring_count = 0
        self.exit_count = 0
        self._init_database()
        self.logger.info("Exit Management Agent initialized")
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Official ADK BaseAgent implementation: Exit management and position monitoring
        
        Args:
            ctx: ADK InvocationContext containing session state and entry recommendations
            
        Yields:
            Event: Progress updates and completion events with exit strategies
        """
        
        try:
            # Extract entry recommendations from previous agent using official state pattern
            entry_recommendations = ctx.session.state.get('entry_candidates', [])
            
            # Yield start event with proper genai types
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Starting exit management for {len(entry_recommendations)} entry recommendations")]
                )
            )
            
            # Generate exit strategies for each recommendation
            exit_strategies = await self.generate_exit_strategies_async(entry_recommendations)
            
            # Yield detailed progress event showing each stock's exit strategy
            if exit_strategies:
                strategy_details = []
                for strategy in exit_strategies[:5]:  # Show first 5 for event display
                    ticker = strategy.get('ticker', 'Unknown')
                    entry = strategy.get('entry_price', 0)
                    stop = strategy.get('stop_loss', 0)
                    target1 = strategy.get('target_1', 0)
                    stop_pct = ((entry - stop) / entry * 100) if entry > 0 else 0
                    target_pct = ((target1 - entry) / entry * 100) if entry > 0 else 0
                    strategy_details.append(f"{ticker}: Entry ₹{entry:.0f}, Stop ₹{stop:.0f} (-{stop_pct:.1f}%), Target ₹{target1:.0f} (+{target_pct:.1f}%)")
                
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part(text=f"Exit strategies generated: {', '.join(strategy_details[:3])}{'...' if len(strategy_details) > 3 else ''}")]
                    )
                )
            
            # Yield progress event
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Generated exit strategies with stop losses and profit targets")]
                )
            )
            
            # Store results in session state (official ADK pattern)
            ctx.session.state['exit_plans'] = exit_strategies
            ctx.session.state['exit_metrics'] = {
                'strategies_generated': len(exit_strategies),
                'monitoring_count': self.monitoring_count,
                'exit_count': self.exit_count
            }
            
            # Yield completion event with structured results
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=
                        f"Exit management complete: {len(exit_strategies)} exit strategies created with risk management parameters"
                    )]
                )
            )
            
        except Exception as e:
            # Yield error event with proper error handling
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Exit management failed: {str(e)}")]
                )
            )
    
    def _init_database(self):
        """Initialize database tables for exit tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Exit history table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS exit_history (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    entry_date TEXT,
                    exit_date TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    exit_reason TEXT,
                    quantity INTEGER,
                    profit_loss REAL,
                    return_pct REAL,
                    holding_days INTEGER,
                    exit_type TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Position status tracking
                conn.execute("""
                CREATE TABLE IF NOT EXISTS position_status (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    current_price REAL,
                    position_return_pct REAL,
                    days_held INTEGER,
                    last_check TEXT,
                    status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                conn.commit()
                self.logger.info("Exit management database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    async def generate_exit_strategies_async(self, entry_recommendations: List[Dict]) -> List[Dict]:
        """
        Generate exit strategies for entry recommendations
        
        Args:
            entry_recommendations: List of entry recommendations from entry timing agent
            
        Returns:
            List of exit strategies with stop losses, targets, and risk management
        """
        try:
            exit_strategies = []
            self.logger.info(f"Generating exit strategies for {len(entry_recommendations)} recommendations")
            
            for recommendation in entry_recommendations:
                try:
                    # Convert entry recommendation to exit strategy
                    exit_strategy = self._convert_to_exit_strategy(recommendation)
                    
                    if exit_strategy:  # Only add valid strategies
                        # Apply exit guardrails
                        is_valid, violations = self.apply_exit_guardrails(exit_strategy)
                        
                        if is_valid:
                            exit_strategies.append(exit_strategy)
                            self.logger.info(f"Exit strategy created for {exit_strategy.get('ticker', 'Unknown')}")
                        else:
                            self.logger.warning(f"Exit strategy rejected for {exit_strategy.get('ticker', 'Unknown')}: {violations}")
                    
                except Exception as e:
                    ticker = recommendation.get('ticker', 'Unknown')
                    self.logger.error(f"Error generating exit strategy for {ticker}: {e}")
                    continue
            
            self.logger.info(f"Generated {len(exit_strategies)} exit strategies")
            return exit_strategies
            
        except Exception as e:
            self.logger.error(f"Exit strategy generation failed: {e}")
            return []
    
    
    def _convert_to_exit_strategy(self, recommendation: Dict) -> Dict:
        """
        Convert entry recommendation to exit strategy
        
        Args:
            recommendation: Entry recommendation from timing agent
            
        Returns:
            Exit strategy with stop losses and profit targets
        """
        try:
            ticker = recommendation.get('ticker', '')
            entry_price = recommendation.get('entry_price', 0)
            risk_metrics = recommendation.get('risk_metrics', {})
            
            # Default position size (can be overridden by actual position management)
            default_quantity = 100  # Default 100 shares for backtesting
            
            return {
                'ticker': ticker,
                'entry_price': entry_price,
                'stop_loss': risk_metrics.get('stop_loss', entry_price * 0.95),
                'hard_sl_price': risk_metrics.get('stop_loss', entry_price * 0.95),  # For guardrails
                'target_1': risk_metrics.get('target_1', entry_price * 1.05),
                'target_2': risk_metrics.get('target_2', entry_price * 1.08),
                'target_3': risk_metrics.get('target_3', entry_price * 1.12),
                'exit_strategy_type': 'SYSTEMATIC',
                'risk_reward_ratio': risk_metrics.get('risk_reward_ratio', 1.5),
                'agent_id': 'exit_management',
                'strategy_timestamp': datetime.now().isoformat(),
                
                # Fields required for guardrails
                'remaining_quantity': default_quantity,
                'current_quantity': default_quantity,    # Current holdings
                'exit_quantity': default_quantity,       # For full exit
                'price_timestamp': datetime.now().isoformat(),
                'exit_type': 'STRATEGY_INIT'
            }
        except Exception as e:
            self.logger.error(f"Error converting {recommendation.get('ticker', 'Unknown')} to exit strategy: {e}")
            return {}
    
    async def _fetch_current_price(self, ticker: str) -> Optional[float]:
        """Fetch current market price for a ticker"""
        try:
            ticker_symbol = f"{ticker}.NS" if not ticker.endswith(('.NS', '.BO')) else ticker
            stock = yf.Ticker(ticker_symbol)
            
            # Get most recent price
            hist = stock.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                return current_price
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Price fetch error for {ticker}: {e}")
            return None
    
    def _store_position_status(self, monitoring_result: Dict):
        """Store position status in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                INSERT INTO position_status 
                (ticker, current_price, position_return_pct, days_held, last_check, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    monitoring_result['ticker'],
                    monitoring_result['current_price'],
                    monitoring_result['position_return_pct'],
                    monitoring_result['days_held'],
                    monitoring_result['last_checked'],
                    monitoring_result['exit_recommendation']
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing position status: {e}")
    
    # 2. EXIT CONDITION CHECKING 
    
    def check_exit_conditions(self, position: Dict, 
                             current_price: float) -> Dict:
        """
        Check all exit conditions and return recommendation
        
        Args:
            position: Position dict with all details
            current_price: Current market price
        
        Returns:
            {
                'should_exit': bool,
                'exit_type': 'SL_HIT' | 'TARGET_1_HIT' | 'TARGET_2_HIT' | 
                            'TARGET_3_HIT' | 'TIME_STOP' | 'HOLD',
                'exit_price': float,
                'exit_reason': str,
                'exit_percent': 0 | 25 | 30 | 40 | 100,  # % of position
                'urgency': 'LOW' | 'MEDIUM' | 'HIGH' | 'IMMEDIATE'
            }
        """
        try:
            # Extract position details
            entry_price = position.get('entry_price', 0)
            hard_sl_price = position.get('hard_sl_price', entry_price * 0.95)
            target_1 = position.get('target_1', entry_price * 1.08)
            target_2 = position.get('target_2', entry_price * 1.12)
            target_3 = position.get('target_3', entry_price * 1.18)
            
            # 1. STOP LOSS HIT (Highest Priority)
            if self.check_sl_condition(position, current_price):
                self.logger.warning(f"Stop loss hit for {position.get('ticker')}: {current_price} <= {hard_sl_price}")
                return {
                    'should_exit': True,
                    'exit_type': 'SL_HIT',
                    'exit_price': hard_sl_price,
                    'exit_reason': f'Stop loss triggered: {current_price} <= {hard_sl_price}',
                    'exit_percent': 100,  # Full exit
                    'urgency': 'IMMEDIATE'
                }
            
            # 2. TIME STOP (120 days) - Second Priority
            if self.check_time_stop_condition(position):
                days_held = self._calculate_days_held(position)
                self.logger.info(f"Time stop condition met for {position.get('ticker')}: {days_held} days")
                return {
                    'should_exit': True,
                    'exit_type': 'TIME_STOP',
                    'exit_price': current_price,
                    'exit_reason': f'Maximum holding period reached: {days_held} days',
                    'exit_percent': 100,  # Full exit
                    'urgency': 'HIGH'
                }
            
            # 3. TARGET CONDITIONS (Order: 3, 2, 1 for highest first)
            target_result = self.check_target_condition(position, current_price)
            if target_result['target_hit']:
                target_level = target_result['target_level']
                target_price = target_result['target_price']
                exit_percent = target_result['exit_percent']
                
                self.logger.info(f"Target {target_level} hit for {position.get('ticker')}: {current_price} >= {target_price}")
                return {
                    'should_exit': True,
                    'exit_type': f'TARGET_{target_level}_HIT',
                    'exit_price': target_price,
                    'exit_reason': f'Target {target_level} achieved: {current_price} >= {target_price}',
                    'exit_percent': exit_percent,
                    'urgency': 'HIGH'
                }
            
            # 4. HOLD (No exit condition met)
            return {
                'should_exit': False,
                'exit_type': 'HOLD',
                'exit_price': current_price,
                'exit_reason': 'No exit conditions met, continue holding',
                'exit_percent': 0,
                'urgency': 'LOW'
            }
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return {
                'should_exit': False,
                'exit_type': 'HOLD',
                'exit_price': current_price,
                'exit_reason': f'Exit condition check failed: {e}',
                'exit_percent': 0,
                'urgency': 'LOW'
            }
    
    def check_sl_condition(self, position: Dict, 
                          current_price: float) -> bool:
        """Check if stop loss is hit"""
        try:
            hard_sl_price = position.get('hard_sl_price')
            if hard_sl_price is None:
                # Calculate default SL if not provided
                entry_price = position.get('entry_price', 0)
                hard_sl_price = entry_price * 0.95  # 5% stop loss
            
            return current_price <= hard_sl_price
            
        except Exception as e:
            self.logger.error(f"Error checking SL condition: {e}")
            return False
    
    def check_target_condition(self, position: Dict, 
                              current_price: float) -> Dict:
        """
        Check if any target is hit
        
        Returns:
            {
                'target_hit': bool,
                'target_level': 1 | 2 | 3,
                'target_price': float,
                'exit_percent': 25 | 30 | 40
            }
        """
        try:
            entry_price = position.get('entry_price', 0)
            
            # Default targets if not provided
            target_1 = position.get('target_1', entry_price * 1.08)  # 8%
            target_2 = position.get('target_2', entry_price * 1.12)  # 12%
            target_3 = position.get('target_3', entry_price * 1.18)  # 18%
            
            # Check targets in order of priority (highest first)
            if current_price >= target_3:
                return {
                    'target_hit': True,
                    'target_level': 3,
                    'target_price': target_3,
                    'exit_percent': 40  # Remaining position
                }
            elif current_price >= target_2:
                return {
                    'target_hit': True,
                    'target_level': 2,
                    'target_price': target_2,
                    'exit_percent': 30  # 30% of remaining after target 1
                }
            elif current_price >= target_1:
                return {
                    'target_hit': True,
                    'target_level': 1,
                    'target_price': target_1,
                    'exit_percent': 25  # 25% of original position
                }
            
            # No target hit
            return {
                'target_hit': False,
                'target_level': 0,
                'target_price': 0,
                'exit_percent': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking target condition: {e}")
            return {
                'target_hit': False,
                'target_level': 0,
                'target_price': 0,
                'exit_percent': 0
            }
    
    def check_time_stop_condition(self, position: Dict) -> bool:
        """Check if 120 days have passed"""
        try:
            days_held = self._calculate_days_held(position)
            return days_held >= 120
            
        except Exception as e:
            self.logger.error(f"Error checking time stop condition: {e}")
            return False
    
    def _calculate_days_held(self, position: Dict) -> int:
        """Calculate number of days position has been held"""
        try:
            entry_date_str = position.get('entry_date', datetime.now().isoformat())
            
            if isinstance(entry_date_str, str):
                try:
                    entry_date = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00'))
                except:
                    entry_date = datetime.now()
            else:
                entry_date = entry_date_str
            
            days_held = (datetime.now() - entry_date).days
            return max(0, days_held)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error calculating days held: {e}")
            return 0
    
    # 3. PARTIAL EXIT LOGIC
    
    def partial_exit_logic(self, position: Dict, 
                          exit_type: str) -> Dict:
        """
        Calculate partial exit quantities (25%-30%-40% scaling)
        
        Portfolio Scaling:
            - Target 1 Hit: Exit 25% of position
            - Target 2 Hit: Exit 30% of remaining position
            - Target 3 Hit: Exit remaining 40%
            - Total: 100% exited across 3 targets
        
        Args:
            position: Full position dict
            exit_type: TARGET_1_HIT | TARGET_2_HIT | TARGET_3_HIT
        
        Returns:
            {
                'remaining_quantity': int,
                'exit_quantity': int,
                'exit_percent': 25 | 30 | 40,
                'exit_value': float,
                'profit_on_this_exit': float,
                'remaining_cost_basis': float
            }
        
        Example:
            Initial position: 1000 shares @ 1000 = 10 Lakhs corpus
            
            Target 1 (25%): Exit 250 shares
                - Remaining: 750 shares
                - Profit: (1100-1000)*250 = 25,000
            
            Target 2 (30% of remaining): Exit 225 shares
                - Remaining: 525 shares
                - Profit: (1200-1000)*225 = 45,000
            
            Target 3 (40% of remaining): Exit 210 shares
                - Remaining: 315 shares
                - Profit: (1250-1000)*210 = 52,500
            
            Final: 315 shares still held or exited based on time stop
        
        Logging:
            Log structured event: partial_exit_calculated
        """
        try:
            # Extract position details
            original_quantity = position.get('quantity', 1000)
            entry_price = position.get('entry_price', 0)
            current_quantity = position.get('current_quantity', original_quantity)  # Track remaining
            
            # Determine exit percentage based on target level
            if exit_type == 'TARGET_1_HIT':
                exit_percent = 25
                exit_quantity = int(original_quantity * 0.25)
            elif exit_type == 'TARGET_2_HIT':
                # 30% of remaining after target 1 (which was 75% of original)
                exit_percent = 30
                exit_quantity = int(current_quantity * 0.30)
            elif exit_type == 'TARGET_3_HIT':
                # 40% of remaining after targets 1 and 2
                exit_percent = 40
                exit_quantity = int(current_quantity * 0.40)
            else:
                # Full exit for SL or time stop
                exit_percent = 100
                exit_quantity = current_quantity
            
            # Calculate remaining quantity
            remaining_quantity = max(0, current_quantity - exit_quantity)
            
            # Calculate exit value and profit
            exit_price = position.get('exit_price', entry_price * 1.1)  # Default 10% gain
            exit_value = exit_quantity * exit_price
            entry_value = exit_quantity * entry_price
            profit_on_this_exit = exit_value - entry_value
            
            # Calculate remaining cost basis
            remaining_cost_basis = remaining_quantity * entry_price
            
            result = {
                'remaining_quantity': remaining_quantity,
                'exit_quantity': exit_quantity,
                'exit_percent': exit_percent,
                'exit_value': round(exit_value, 2),
                'profit_on_this_exit': round(profit_on_this_exit, 2),
                'remaining_cost_basis': round(remaining_cost_basis, 2),
                'original_quantity': original_quantity,
                'current_quantity': current_quantity,
                'exit_type': exit_type
            }
            
            self.logger.info(f"Partial exit calculated for {position.get('ticker', 'UNKNOWN')}: "
                           f"Exit {exit_quantity} shares ({exit_percent}%), "
                           f"Profit: ₹{profit_on_this_exit:,.0f}, "
                           f"Remaining: {remaining_quantity} shares")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating partial exit: {e}")
            return {
                'remaining_quantity': 0,
                'exit_quantity': 0,
                'exit_percent': 0,
                'exit_value': 0.0,
                'profit_on_this_exit': 0.0,
                'remaining_cost_basis': 0.0
            }
    
    # 4. EXIT GUARDRAILS
    
    def apply_exit_guardrails(self, exit_decision: Dict) -> Tuple[bool, List[str]]:
        """
        Apply 5 safety guardrails before exit execution
        
        Guardrail 1: Never exit same position twice
            Check: remaining_quantity > 0
        
        Guardrail 2: Don't execute if price is stale (> 5 min old)
            Check: (now - price_timestamp).seconds < 300
        
        Guardrail 3: SL must be between 0 and entry_price
            Check: 0 < hard_sl_price < entry_price
        
        Guardrail 4: Target prices must be > entry_price
            Check: all_targets > entry_price
        
        Guardrail 5: Partial exit quantity > 0 and <= remaining
            Check: 0 < exit_quantity <= remaining_quantity
        
        Returns:
            (passes_all: bool, violations: List[str])
        
        Logic:
            violations = []
            
            if not check_guardrail_1():
                violations.append("Position already fully exited")
            
            if not check_guardrail_2():
                violations.append("Price data is stale (>5 min)")
            
            if not check_guardrail_3():
                violations.append("SL outside valid range")
            
            if not check_guardrail_4():
                violations.append("Target below entry price")
            
            if not check_guardrail_5():
                violations.append("Exit quantity invalid")
            
            if violations:
                log_guardrail_violations(violations)
                return False, violations
            
            return True, []
        
        Logging:
            Log structured event: exit_guardrails_checked
            Log violations to guardrail_violations table
        """
        try:
            violations = []
            ticker = exit_decision.get('ticker', 'UNKNOWN')
            
            # Guardrail 1: Never exit same position twice
            remaining_quantity = exit_decision.get('remaining_quantity', 0)
            if remaining_quantity <= 0 and exit_decision.get('exit_type') != 'SL_HIT':
                violations.append("Position already fully exited or invalid remaining quantity")
            
            # Guardrail 2: Don't execute if price is stale (> 5 min old)
            price_timestamp_str = exit_decision.get('price_timestamp')
            if price_timestamp_str:
                try:
                    price_timestamp = datetime.fromisoformat(price_timestamp_str.replace('Z', '+00:00'))
                    time_diff = (datetime.now() - price_timestamp).total_seconds()
                    if time_diff > 300:  # 5 minutes
                        violations.append(f"Price data is stale: {time_diff:.0f} seconds old (>300)")
                except Exception:
                    violations.append("Invalid price timestamp format")
            
            # Guardrail 3: SL must be between 0 and entry_price
            entry_price = exit_decision.get('entry_price', 0)
            hard_sl_price = exit_decision.get('hard_sl_price', 0)
            
            if hard_sl_price <= 0:
                violations.append(f"Stop loss price invalid: {hard_sl_price}")
            elif hard_sl_price >= entry_price:
                violations.append(f"Stop loss above entry price: {hard_sl_price} >= {entry_price}")
            
            # Guardrail 4: Target prices must be > entry_price
            target_1 = exit_decision.get('target_1', entry_price * 1.08)
            target_2 = exit_decision.get('target_2', entry_price * 1.12)
            target_3 = exit_decision.get('target_3', entry_price * 1.18)
            
            if target_1 <= entry_price:
                violations.append(f"Target 1 below entry price: {target_1} <= {entry_price}")
            if target_2 <= entry_price:
                violations.append(f"Target 2 below entry price: {target_2} <= {entry_price}")
            if target_3 <= entry_price:
                violations.append(f"Target 3 below entry price: {target_3} <= {entry_price}")
            
            # Guardrail 5: Partial exit quantity > 0 and <= remaining
            exit_quantity = exit_decision.get('exit_quantity', 0)
            current_quantity = exit_decision.get('current_quantity', 0)
            
            if exit_quantity <= 0:
                violations.append(f"Exit quantity invalid: {exit_quantity}")
            elif exit_quantity > current_quantity:
                violations.append(f"Exit quantity exceeds holdings: {exit_quantity} > {current_quantity}")
            
            # Log violations if any
            if violations:
                self._log_exit_guardrail_violations(ticker, violations)
                self.logger.warning(f"Exit guardrail violations for {ticker}: {violations}")
                return False, violations
            
            self.logger.info(f"All exit guardrails passed for {ticker}")
            return True, []
            
        except Exception as e:
            self.logger.error(f"Error applying exit guardrails: {e}")
            return False, [f"Guardrail system error: {e}"]
    
    def _log_exit_guardrail_violations(self, ticker: str, violations: List[str]):
        """Log exit guardrail violations to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for violation in violations:
                    conn.execute("""
                    INSERT INTO guardrail_violations (ticker, violation_type, details)
                    VALUES (?, ?, ?)
                    """, (ticker, 'EXIT_GUARDRAIL', violation))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging exit guardrail violations: {e}")
    
    # 5. P&L TRACKING 
    
    def calculate_pnl(self, position: Dict, 
                     exit_price: float, 
                     exit_quantity: int) -> Dict:
        """
        Calculate profit/loss on exit
        
        Returns:
            {
                'entry_value': float,      # entry_price * quantity
                'exit_value': float,       # exit_price * exit_quantity
                'gross_profit': float,     # exit_value - entry_value
                'return_pct': float,       # (profit / entry_value) * 100
                'holding_days': int,
                'daily_return_pct': float, # return_pct / holding_days
                'exit_commission': float,  # Estimated broker fee
                'net_profit': float        # gross_profit - commission
            }
        
        Logging:
            Log structured event: pnl_calculated, return_pct, holding_days
        """
        try:
            # Extract position details
            entry_price = position.get('entry_price', 0)
            entry_date_str = position.get('entry_date', datetime.now().isoformat())
            
            # Calculate holding period
            holding_days = self._calculate_days_held(position)
            
            # Calculate P&L
            entry_value = entry_price * exit_quantity
            exit_value = exit_price * exit_quantity
            gross_profit = exit_value - entry_value
            
            # Calculate return percentage
            return_pct = (gross_profit / entry_value * 100) if entry_value > 0 else 0
            
            # Calculate daily return
            daily_return_pct = (return_pct / holding_days) if holding_days > 0 else 0
            
            # Estimate commission (0.1% of exit value - typical brokerage)
            exit_commission = exit_value * 0.001  # 0.1% 
            
            # Net profit after commission
            net_profit = gross_profit - exit_commission
            
            pnl_result = {
                'entry_value': round(entry_value, 2),
                'exit_value': round(exit_value, 2),
                'gross_profit': round(gross_profit, 2),
                'return_pct': round(return_pct, 2),
                'holding_days': holding_days,
                'daily_return_pct': round(daily_return_pct, 4),
                'exit_commission': round(exit_commission, 2),
                'net_profit': round(net_profit, 2),
                'exit_price': exit_price,
                'exit_quantity': exit_quantity,
                'entry_price': entry_price
            }
            
            self.logger.info(f"P&L calculated for {position.get('ticker', 'UNKNOWN')}: "
                           f"Return: {return_pct:+.2f}% over {holding_days} days, "
                           f"Net Profit: ₹{net_profit:,.0f}")
            
            return pnl_result
            
        except Exception as e:
            self.logger.error(f"Error calculating P&L: {e}")
            return {
                'entry_value': 0.0, 'exit_value': 0.0, 'gross_profit': 0.0,
                'return_pct': 0.0, 'holding_days': 0, 'daily_return_pct': 0.0,
                'exit_commission': 0.0, 'net_profit': 0.0
            }
    
    def track_exit_history(self, position: Dict, 
                          exit_data: Dict) -> None:
        """
        Track exit in database for analysis and memory
        
        Query:
            INSERT INTO exit_history 
            (ticker, entry_date, exit_date, entry_price, exit_price,
             exit_reason, quantity, profit_loss, return_pct, holding_days)
            VALUES (...)
        
        Also update memory_declarative with success metrics
        
        Logging:
            Log structured event: exit_tracked, ticker, profit_loss
        """
        try:
            ticker = position.get('ticker', 'UNKNOWN')
            
            # Extract data
            entry_date = position.get('entry_date', datetime.now().isoformat())
            exit_date = datetime.now().isoformat()
            entry_price = exit_data.get('entry_price', 0)
            exit_price = exit_data.get('exit_price', 0)
            exit_reason = exit_data.get('exit_type', 'MANUAL')
            quantity = exit_data.get('exit_quantity', 0)
            profit_loss = exit_data.get('net_profit', 0)
            return_pct = exit_data.get('return_pct', 0)
            holding_days = exit_data.get('holding_days', 0)
            
            # Store in exit_history table
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                INSERT INTO exit_history 
                (ticker, entry_date, exit_date, entry_price, exit_price,
                 exit_reason, quantity, profit_loss, return_pct, holding_days, exit_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, entry_date, exit_date, entry_price, exit_price,
                    exit_reason, quantity, profit_loss, return_pct, holding_days, exit_reason
                ))
                
                # Update memory with success pattern if profitable
                if return_pct > 5:  # Profitable exit
                    success_pattern = f"exit_{exit_reason.lower()}_{holding_days}days"
                    confidence = min(0.9, return_pct / 20)  # Scale confidence
                    
                    conn.execute("""
                    INSERT OR REPLACE INTO memory_declarative
                    (ticker, pattern, confidence, success_rate, last_used)
                    VALUES (?, ?, ?, ?, ?)
                    """, (
                        ticker, success_pattern, confidence, 
                        return_pct / 100, datetime.now().isoformat()
                    ))
                
                conn.commit()
            
            self.logger.info(f"Exit tracked for {ticker}: {exit_reason}, "
                           f"P&L: ₹{profit_loss:,.0f}, Return: {return_pct:+.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error tracking exit history: {e}")
    
    def _calculate_days_held(self, position: Dict) -> int:
        """Calculate number of days position has been held"""
        try:
            entry_date_str = position.get('entry_date', datetime.now().isoformat())
            if isinstance(entry_date_str, str):
                entry_date = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00'))
            else:
                entry_date = entry_date_str
            
            days_held = (datetime.now() - entry_date).days
            return max(0, days_held)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate days held: {e}")
            return 0
    
    # 6. MAIN MANAGEMENT METHOD
    
    def manage_position(self, position: Dict) -> Dict:
        """
        Single position management (called by orchestrator)
        
        Pipeline:
            1. Fetch current price
            2. Check exit conditions
            3. Apply guardrails
            4. Calculate P&L for partial exits
            5. Execute exit
            6. Track in database
            7. Update memory
        """
        ticker = position.get('ticker', 'UNKNOWN')
        
        try:
            self.logger.info(f"Managing position for {ticker}")
            
            # Step 1: Fetch current price
            current_price = asyncio.run(self._fetch_current_price(ticker))
            if current_price is None:
                return {
                    'ticker': ticker, 'action': 'ERROR', 'success': False,
                    'error': 'Could not fetch current price'
                }
            
            # Step 2: Check exit conditions
            exit_condition = self.check_exit_conditions(position, current_price)
            
            if not exit_condition['should_exit']:
                # HOLD - no exit needed
                days_held = self._calculate_days_held(position)
                entry_price = position.get('entry_price', 0)
                return_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                
                return {
                    'ticker': ticker, 'action': 'HOLD', 'exit_type': 'HOLD',
                    'return_pct': return_pct, 'holding_days': days_held, 'success': True
                }
            
            # Exit processing logic continues...
            exit_type = exit_condition['exit_type']
            exit_price = exit_condition['exit_price']
            
            if 'TARGET' in exit_type:
                partial_exit_data = self.partial_exit_logic(position, exit_type)
                exit_quantity = partial_exit_data['exit_quantity']
                action = 'PARTIAL_EXIT'
            else:
                exit_quantity = position.get('quantity', 0)
                action = 'FULL_EXIT'
            
            # Calculate P&L and track
            pnl_data = self.calculate_pnl(position, exit_price, exit_quantity)
            exit_data = {**pnl_data, 'exit_type': exit_type}
            self.track_exit_history(position, exit_data)
            
            result = {
                'ticker': ticker, 'action': action, 'exit_type': exit_type,
                'exit_price': exit_price, 'exit_quantity': exit_quantity,
                'profit_loss': pnl_data['net_profit'], 'return_pct': pnl_data['return_pct'],
                'holding_days': pnl_data['holding_days'], 'success': True
            }
            
            self.exit_count += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Error managing position for {ticker}: {e}")
            return {'ticker': ticker, 'action': 'ERROR', 'success': False, 'error': str(e)}
    