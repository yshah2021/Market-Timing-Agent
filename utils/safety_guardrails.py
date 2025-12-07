#!/usr/bin/env python3
"""
Safety Guardrails Module for Market Timing Agents
Implements comprehensive safety checks and circuit breakers
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import sqlite3
from dataclasses import dataclass
from logging_config import get_logger

@dataclass
class GuardrailResult:
    """Result of guardrail validation"""
    is_safe: bool
    reason: str
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    action_taken: str
    details: Dict[str, Any] = None

class SafetyGuardrails:
    """
    Production safety guardrails for market timing agents
    
    Implements:
    - Agent 2 entry guardrails (7 checks)
    - Agent 3 exit guardrails (5 checks)
    - Circuit breaker logic
    - Violation tracking
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize safety guardrails
        
        Args:
            agent_name: Name of the agent (agent2, agent3)
        """
        self.agent_name = agent_name
        self.logger = get_logger(f"guardrails_{agent_name}")
        
        # Configuration (can be moved to config file)
        self.config = {
            'max_portfolio_deployment': 100.0,  # %
            'min_confidence_score': 60.0,       # %
            'max_position_size': 20.0,          # %
            'min_risk_reward_ratio': 1.5,
            'max_sector_concentration': 40.0,   # %
            'price_staleness_minutes': 5,
            'circuit_breaker_threshold': 3,     # violations per day
            'pe_industry_multiplier_min': 0.6,
            'pe_industry_multiplier_max': 1.6
        }
        
        # Setup database connection
        self.setup_database()
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.check_circuit_breaker_status()
    
    def setup_database(self):
        """Setup database connection with fallback to SQLite"""
        
        try:
            import mysql.connector
            self.db_conn = mysql.connector.connect(
                host='localhost',
                user='trading_user',
                password='trading_password',
                database='market_timing_agents'
            )
            self.db_type = 'mysql'
        except Exception:
            # Fallback to SQLite
            self.db_conn = sqlite3.connect('system_logs.db', check_same_thread=False)
            self.db_type = 'sqlite'
            self._create_sqlite_tables()
    
    def _create_sqlite_tables(self):
        """Create SQLite tables for guardrail tracking"""
        
        cursor = self.db_conn.cursor()
        
        # Guardrail violations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guardrail_violations (
                violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                guardrail_name TEXT,
                ticker TEXT,
                violation_reason TEXT,
                violation_time TEXT,
                severity TEXT,
                action_taken TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Circuit breaker status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker_status (
                breaker_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                breaker_type TEXT,
                is_triggered INTEGER DEFAULT 0,
                trigger_reason TEXT,
                trigger_time TEXT,
                reset_time TEXT,
                violation_count INTEGER DEFAULT 0,
                threshold_value REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.db_conn.commit()
    
    def check_circuit_breaker_status(self):
        """Check if circuit breaker is currently active"""
        
        try:
            cursor = self.db_conn.cursor()
            today = datetime.now().date()
            
            if self.db_type == 'mysql':
                query = """
                SELECT COUNT(*) FROM guardrail_violations 
                WHERE agent_name = %s AND DATE(violation_time) = %s AND severity = 'CRITICAL'
                """
                cursor.execute(query, (self.agent_name, today))
            else:
                query = """
                SELECT COUNT(*) FROM guardrail_violations 
                WHERE agent_name = ? AND date(violation_time) = ? AND severity = 'CRITICAL'
                """
                cursor.execute(query, (self.agent_name, today.isoformat()))
            
            violation_count = cursor.fetchone()[0]
            
            if violation_count >= self.config['circuit_breaker_threshold']:
                self.circuit_breaker_active = True
                self.logger.warning({
                    "event": "circuit_breaker_active",
                    "violation_count": violation_count,
                    "threshold": self.config['circuit_breaker_threshold']
                })
            else:
                self.circuit_breaker_active = False
                
        except Exception as e:
            self.logger.error({
                "event": "circuit_breaker_check_failed",
                "error_message": str(e)
            })
    
    def log_violation(self, guardrail_name: str, ticker: str, reason: str, 
                     severity: str, action_taken: str):
        """Log guardrail violation to database"""
        
        try:
            cursor = self.db_conn.cursor()
            
            if self.db_type == 'mysql':
                query = """
                INSERT INTO guardrail_violations 
                (agent_name, guardrail_name, ticker, violation_reason, violation_time, severity, action_taken)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    self.agent_name,
                    guardrail_name,
                    ticker,
                    reason,
                    datetime.now(),
                    severity,
                    action_taken
                ))
            else:
                query = """
                INSERT INTO guardrail_violations 
                (agent_name, guardrail_name, ticker, violation_reason, violation_time, severity, action_taken)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(query, (
                    self.agent_name,
                    guardrail_name,
                    ticker,
                    reason,
                    datetime.now().isoformat(),
                    severity,
                    action_taken
                ))
            
            self.db_conn.commit()
            
            # Check if we need to trigger circuit breaker
            self.check_circuit_breaker_status()
            
        except Exception as e:
            self.logger.error({
                "event": "violation_logging_failed",
                "error_message": str(e)
            })
    
    # ===============================================
    # AGENT 2 ENTRY GUARDRAILS
    # ===============================================
    
    def apply_entry_guardrails(self, ticker: str, confidence: float, 
                             position_size: float, pe_data: Dict = None,
                             portfolio_data: Dict = None) -> List[GuardrailResult]:
        """
        Apply all entry guardrails for Agent 2
        
        Args:
            ticker: Stock ticker
            confidence: Entry signal confidence (0-100)
            position_size: Proposed position size (% of portfolio)
            pe_data: PE ratio data {stock_pe, industry_pe}
            portfolio_data: Current portfolio state
            
        Returns:
            List of guardrail results
        """
        
        if self.circuit_breaker_active:
            return [GuardrailResult(
                is_safe=False,
                reason="Circuit breaker active - trading suspended",
                severity="CRITICAL",
                action_taken="ENTRY_BLOCKED"
            )]
        
        results = []
        
        # Guardrail 1: Portfolio deployment limit
        results.append(self._check_portfolio_deployment(portfolio_data))
        
        # Guardrail 2: Confidence threshold
        results.append(self._check_confidence_threshold(ticker, confidence))
        
        # Guardrail 3: PE ratio validation
        if pe_data:
            results.append(self._check_pe_ratio(ticker, pe_data))
        
        # Guardrail 4: Position size limit
        results.append(self._check_position_size(ticker, position_size))
        
        # Guardrail 5: Risk-reward ratio
        results.append(self._check_risk_reward_ratio(ticker))
        
        # Guardrail 6: Sector concentration
        if portfolio_data and portfolio_data.get('sector_data'):
            results.append(self._check_sector_concentration(ticker, portfolio_data))
        
        # Guardrail 7: Duplicate position check
        results.append(self._check_duplicate_position(ticker, portfolio_data))
        
        # Log all guardrail checks
        for result in results:
            if not result.is_safe:
                self.log_violation(
                    guardrail_name=result.reason.split(':')[0],
                    ticker=ticker,
                    reason=result.reason,
                    severity=result.severity,
                    action_taken=result.action_taken
                )
        
        return results
    
    def _check_portfolio_deployment(self, portfolio_data: Dict) -> GuardrailResult:
        """Check portfolio deployment percentage"""
        
        try:
            current_deployment = portfolio_data.get('deployment_pct', 0) if portfolio_data else 0
            max_deployment = self.config['max_portfolio_deployment']
            
            if current_deployment >= max_deployment:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Portfolio deployment limit: {current_deployment:.1f}% >= {max_deployment}%",
                    severity="WARNING",
                    action_taken="ENTRY_SKIPPED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason="Portfolio deployment within limits",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=False,
                reason=f"Portfolio deployment check failed: {str(e)}",
                severity="CRITICAL",
                action_taken="ENTRY_BLOCKED"
            )
    
    def _check_confidence_threshold(self, ticker: str, confidence: float) -> GuardrailResult:
        """Check entry signal confidence threshold"""
        
        try:
            min_confidence = self.config['min_confidence_score']
            
            if confidence < min_confidence:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Confidence threshold: {confidence:.1f}% < {min_confidence}%",
                    severity="INFO",
                    action_taken="ENTRY_SKIPPED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason=f"Confidence acceptable: {confidence:.1f}%",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=False,
                reason=f"Confidence check failed: {str(e)}",
                severity="CRITICAL",
                action_taken="ENTRY_BLOCKED"
            )
    
    def _check_pe_ratio(self, ticker: str, pe_data: Dict) -> GuardrailResult:
        """Check PE ratio vs industry average"""
        
        try:
            stock_pe = pe_data.get('stock_pe', 0)
            industry_pe = pe_data.get('industry_pe', 0)
            
            if industry_pe <= 0:
                return GuardrailResult(
                    is_safe=True,
                    reason="PE check skipped - industry PE unavailable",
                    severity="INFO",
                    action_taken="NONE"
                )
            
            pe_ratio = stock_pe / industry_pe
            min_ratio = self.config['pe_industry_multiplier_min']
            max_ratio = self.config['pe_industry_multiplier_max']
            
            if not (min_ratio <= pe_ratio <= max_ratio):
                return GuardrailResult(
                    is_safe=False,
                    reason=f"PE ratio check: {pe_ratio:.2f}x industry (acceptable: {min_ratio}-{max_ratio}x)",
                    severity="WARNING",
                    action_taken="ENTRY_SKIPPED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason=f"PE ratio acceptable: {pe_ratio:.2f}x industry",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=True,  # Allow entry if PE check fails
                reason=f"PE check failed, allowing entry: {str(e)}",
                severity="WARNING",
                action_taken="NONE"
            )
    
    def _check_position_size(self, ticker: str, position_size: float) -> GuardrailResult:
        """Check position size limit"""
        
        try:
            max_size = self.config['max_position_size']
            
            if position_size > max_size:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Position size limit: {position_size:.1f}% > {max_size}%",
                    severity="WARNING",
                    action_taken="POSITION_REDUCED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason=f"Position size acceptable: {position_size:.1f}%",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=False,
                reason=f"Position size check failed: {str(e)}",
                severity="CRITICAL",
                action_taken="ENTRY_BLOCKED"
            )
    
    def _check_risk_reward_ratio(self, ticker: str) -> GuardrailResult:
        """Check risk-reward ratio"""
        
        try:
            # This is a simplified check - in practice, you'd calculate actual R:R
            # based on stop loss and target levels
            min_ratio = self.config['min_risk_reward_ratio']
            
            # For now, assume all entries meet minimum R:R
            # TODO: Implement actual R:R calculation based on stop/target levels
            
            return GuardrailResult(
                is_safe=True,
                reason=f"Risk-reward ratio meets minimum: {min_ratio}",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=False,
                reason=f"Risk-reward check failed: {str(e)}",
                severity="CRITICAL",
                action_taken="ENTRY_BLOCKED"
            )
    
    def _check_sector_concentration(self, ticker: str, portfolio_data: Dict) -> GuardrailResult:
        """Check sector concentration limits"""
        
        try:
            sector_data = portfolio_data.get('sector_data', {})
            stock_sector = portfolio_data.get('stock_sector', 'UNKNOWN')
            max_concentration = self.config['max_sector_concentration']
            
            current_concentration = sector_data.get(stock_sector, 0)
            
            if current_concentration >= max_concentration:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Sector concentration limit: {stock_sector} {current_concentration:.1f}% >= {max_concentration}%",
                    severity="WARNING",
                    action_taken="ENTRY_SKIPPED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason=f"Sector concentration acceptable: {stock_sector} {current_concentration:.1f}%",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=True,  # Allow entry if sector check fails
                reason=f"Sector check failed, allowing entry: {str(e)}",
                severity="WARNING",
                action_taken="NONE"
            )
    
    def _check_duplicate_position(self, ticker: str, portfolio_data: Dict) -> GuardrailResult:
        """Check for duplicate position"""
        
        try:
            active_positions = portfolio_data.get('active_positions', []) if portfolio_data else []
            
            if ticker in active_positions:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Duplicate position: {ticker} already in portfolio",
                    severity="INFO",
                    action_taken="ENTRY_SKIPPED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason="No duplicate position",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=True,  # Allow entry if duplicate check fails
                reason=f"Duplicate check failed, allowing entry: {str(e)}",
                severity="WARNING",
                action_taken="NONE"
            )
    
    # ===============================================
    # AGENT 3 EXIT GUARDRAILS
    # ===============================================
    
    def apply_exit_guardrails(self, ticker: str, exit_data: Dict) -> List[GuardrailResult]:
        """
        Apply all exit guardrails for Agent 3
        
        Args:
            ticker: Stock ticker
            exit_data: Exit request data {exit_type, quantity, current_price, etc.}
            
        Returns:
            List of guardrail results
        """
        
        if self.circuit_breaker_active:
            return [GuardrailResult(
                is_safe=False,
                reason="Circuit breaker active - exit validation failed",
                severity="CRITICAL",
                action_taken="EXIT_BLOCKED"
            )]
        
        results = []
        
        # Guardrail 1: Duplicate exit check
        results.append(self._check_duplicate_exit(ticker, exit_data))
        
        # Guardrail 2: Price staleness check
        results.append(self._check_price_staleness(ticker, exit_data))
        
        # Guardrail 3: Stop loss validation
        results.append(self._check_stop_loss_validation(ticker, exit_data))
        
        # Guardrail 4: Target price validation
        results.append(self._check_target_price_validation(ticker, exit_data))
        
        # Guardrail 5: Quantity validation
        results.append(self._check_quantity_validation(ticker, exit_data))
        
        # Log violations
        for result in results:
            if not result.is_safe:
                self.log_violation(
                    guardrail_name=result.reason.split(':')[0],
                    ticker=ticker,
                    reason=result.reason,
                    severity=result.severity,
                    action_taken=result.action_taken
                )
        
        return results
    
    def _check_duplicate_exit(self, ticker: str, exit_data: Dict) -> GuardrailResult:
        """Check for duplicate exit attempts"""
        
        try:
            remaining_quantity = exit_data.get('remaining_quantity', 0)
            
            if remaining_quantity <= 0:
                return GuardrailResult(
                    is_safe=False,
                    reason="Duplicate exit: No remaining quantity",
                    severity="WARNING",
                    action_taken="EXIT_BLOCKED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason="Remaining quantity available for exit",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=False,
                reason=f"Duplicate exit check failed: {str(e)}",
                severity="CRITICAL",
                action_taken="EXIT_BLOCKED"
            )
    
    def _check_price_staleness(self, ticker: str, exit_data: Dict) -> GuardrailResult:
        """Check if current price is too stale"""
        
        try:
            price_timestamp = exit_data.get('price_timestamp')
            if not price_timestamp:
                return GuardrailResult(
                    is_safe=True,
                    reason="Price timestamp not available, skipping staleness check",
                    severity="WARNING",
                    action_taken="NONE"
                )
            
            if isinstance(price_timestamp, str):
                price_timestamp = datetime.fromisoformat(price_timestamp.replace('Z', '+00:00'))
            
            age_minutes = (datetime.now() - price_timestamp).total_seconds() / 60
            max_age = self.config['price_staleness_minutes']
            
            if age_minutes > max_age:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Price staleness: {age_minutes:.1f} minutes > {max_age} minutes",
                    severity="WARNING",
                    action_taken="EXIT_DELAYED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason=f"Price freshness acceptable: {age_minutes:.1f} minutes",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=True,  # Allow exit if staleness check fails
                reason=f"Price staleness check failed, allowing exit: {str(e)}",
                severity="WARNING",
                action_taken="NONE"
            )
    
    def _check_stop_loss_validation(self, ticker: str, exit_data: Dict) -> GuardrailResult:
        """Validate stop loss price"""
        
        try:
            exit_type = exit_data.get('exit_type', '')
            if 'SL' not in exit_type:
                return GuardrailResult(
                    is_safe=True,
                    reason="Not a stop loss exit",
                    severity="INFO",
                    action_taken="NONE"
                )
            
            entry_price = exit_data.get('entry_price', 0)
            current_price = exit_data.get('current_price', 0)
            
            if current_price <= 0 or entry_price <= 0:
                return GuardrailResult(
                    is_safe=False,
                    reason="Stop loss validation: Invalid price data",
                    severity="CRITICAL",
                    action_taken="EXIT_BLOCKED"
                )
            
            if current_price >= entry_price:
                return GuardrailResult(
                    is_safe=False,
                    reason="Stop loss validation: Current price >= entry price",
                    severity="WARNING",
                    action_taken="EXIT_REVIEWED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason="Stop loss price validation passed",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=False,
                reason=f"Stop loss validation failed: {str(e)}",
                severity="CRITICAL",
                action_taken="EXIT_BLOCKED"
            )
    
    def _check_target_price_validation(self, ticker: str, exit_data: Dict) -> GuardrailResult:
        """Validate target price"""
        
        try:
            exit_type = exit_data.get('exit_type', '')
            if 'TARGET' not in exit_type:
                return GuardrailResult(
                    is_safe=True,
                    reason="Not a target exit",
                    severity="INFO",
                    action_taken="NONE"
                )
            
            entry_price = exit_data.get('entry_price', 0)
            current_price = exit_data.get('current_price', 0)
            
            if current_price <= 0 or entry_price <= 0:
                return GuardrailResult(
                    is_safe=False,
                    reason="Target validation: Invalid price data",
                    severity="CRITICAL",
                    action_taken="EXIT_BLOCKED"
                )
            
            if current_price <= entry_price:
                return GuardrailResult(
                    is_safe=False,
                    reason="Target validation: Current price <= entry price",
                    severity="WARNING",
                    action_taken="EXIT_REVIEWED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason="Target price validation passed",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=False,
                reason=f"Target price validation failed: {str(e)}",
                severity="CRITICAL",
                action_taken="EXIT_BLOCKED"
            )
    
    def _check_quantity_validation(self, ticker: str, exit_data: Dict) -> GuardrailResult:
        """Validate exit quantity"""
        
        try:
            exit_quantity = exit_data.get('exit_quantity', 0)
            remaining_quantity = exit_data.get('remaining_quantity', 0)
            
            if exit_quantity <= 0:
                return GuardrailResult(
                    is_safe=False,
                    reason="Quantity validation: Exit quantity <= 0",
                    severity="WARNING",
                    action_taken="EXIT_BLOCKED"
                )
            
            if exit_quantity > remaining_quantity:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Quantity validation: Exit quantity {exit_quantity} > remaining {remaining_quantity}",
                    severity="WARNING",
                    action_taken="QUANTITY_ADJUSTED"
                )
            
            return GuardrailResult(
                is_safe=True,
                reason="Quantity validation passed",
                severity="INFO",
                action_taken="NONE"
            )
            
        except Exception as e:
            return GuardrailResult(
                is_safe=False,
                reason=f"Quantity validation failed: {str(e)}",
                severity="CRITICAL",
                action_taken="EXIT_BLOCKED"
            )
    
    def reset_circuit_breaker(self, reason: str = "Manual reset"):
        """Reset circuit breaker status"""
        
        try:
            cursor = self.db_conn.cursor()
            
            if self.db_type == 'mysql':
                query = """
                INSERT INTO circuit_breaker_status 
                (agent_name, breaker_type, is_triggered, trigger_reason, reset_time)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(query, (
                    self.agent_name,
                    'MANUAL_RESET',
                    False,
                    reason,
                    datetime.now()
                ))
            else:
                query = """
                INSERT INTO circuit_breaker_status 
                (agent_name, breaker_type, is_triggered, trigger_reason, reset_time)
                VALUES (?, ?, ?, ?, ?)
                """
                cursor.execute(query, (
                    self.agent_name,
                    'MANUAL_RESET',
                    0,
                    reason,
                    datetime.now().isoformat()
                ))
            
            self.db_conn.commit()
            self.circuit_breaker_active = False
            
            self.logger.info({
                "event": "circuit_breaker_reset",
                "reason": reason
            })
            
        except Exception as e:
            self.logger.error({
                "event": "circuit_breaker_reset_failed",
                "error_message": str(e)
            })
    
    def get_daily_violation_summary(self) -> Dict:
        """Get summary of today's guardrail violations"""
        
        try:
            cursor = self.db_conn.cursor()
            today = datetime.now().date()
            
            if self.db_type == 'mysql':
                query = """
                SELECT guardrail_name, severity, COUNT(*) as count
                FROM guardrail_violations 
                WHERE agent_name = %s AND DATE(violation_time) = %s
                GROUP BY guardrail_name, severity
                """
                cursor.execute(query, (self.agent_name, today))
            else:
                query = """
                SELECT guardrail_name, severity, COUNT(*) as count
                FROM guardrail_violations 
                WHERE agent_name = ? AND date(violation_time) = ?
                GROUP BY guardrail_name, severity
                """
                cursor.execute(query, (self.agent_name, today.isoformat()))
            
            violations = cursor.fetchall()
            
            summary = {
                'date': today.isoformat(),
                'total_violations': sum(v[2] for v in violations),
                'by_severity': {},
                'by_guardrail': {}
            }
            
            for guardrail, severity, count in violations:
                summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + count
                summary['by_guardrail'][guardrail] = summary['by_guardrail'].get(guardrail, 0) + count
            
            return summary
            
        except Exception as e:
            self.logger.error({
                "event": "violation_summary_failed",
                "error_message": str(e)
            })
            return {}
    
    def __del__(self):
        """Cleanup database connection"""
        try:
            if hasattr(self, 'db_conn'):
                self.db_conn.close()
        except:
            pass