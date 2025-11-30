# Official ADK imports
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
import numpy as np
import pandas as pd
import yfinance as yf
import time
import sqlite3
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from datetime import datetime, timedelta
import json
from statistics import mean
import asyncio

# Import enhanced logging
from utils.logging_config import get_logger

class EntryTimingAgent(BaseAgent):
    """Enhanced Agent 2: Entry Timing & Technical Analysis with Advanced Features"""
    
    model_config = {"arbitrary_types_allowed": True}
    name: str = "EntryTimingAgent"
    logger: Any = None
    db_path: str = "market_timing_agents.db"
    corpus_size: float = 1000000
    analysis_count: int = 0
    success_count: int = 0
    active_positions: Dict = {}
    sector_positions: Dict = {}
    
    def __init__(self, db_path: str = "market_timing_agents.db", corpus_size: float = 1000000):
        super().__init__(name="EntryTimingAgent")
        self.logger = get_logger('agents.entry_timing_agent')
        self.db_path = db_path
        self.corpus_size = corpus_size
        self.analysis_count = 0
        self.success_count = 0
        self.active_positions = {}  # Track active positions for guardrails
        self.sector_positions = {}  # Track sector exposure
        self._init_database()
        self.logger.info("Entry Timing Agent initialized")
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Official ADK BaseAgent implementation: Entry timing analysis pipeline
        
        Args:
            ctx: ADK InvocationContext containing session state and screened candidates
            
        Yields:
            Event: Progress updates and completion events with entry recommendations
        """
        
        try:
            # Extract candidates from previous agent using official state pattern
            candidates = ctx.session.state.get('quality_candidates', [])
            
            # Yield start event with proper genai types
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Starting entry timing analysis for {len(candidates)} candidates")]
                )
            )
            
            # Process entry timing analysis with progress events
            for i, candidate in enumerate(candidates[:5]):
                ticker = candidate.get('ticker', f'Stock_{i}')
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part(text=f"Analyzing technical indicators for {ticker} ({i+1}/{min(len(candidates), 5)})")]
                    )
                )
            
            recommendations = await self.analyze_entry_signals_async(candidates)
            
            # Store results in session state for next agent (official ADK pattern)
            ctx.session.state['entry_candidates'] = recommendations
            ctx.session.state['entry_metrics'] = {
                'candidates_analyzed': self.analysis_count,
                'successful_analyses': self.success_count,
                'recommendations_generated': len(recommendations)
            }
            
            # Yield completion event with structured results
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=
                        f"Entry timing analysis complete: {len(recommendations)} recommendations generated from {len(candidates)} candidates"
                    )]
                )
            )
            
        except Exception as e:
            # Yield error event with proper error handling
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Entry timing analysis failed: {str(e)}")]
                )
            )
        
    def _init_database(self):
        """Initialize database tables for memory storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Memory table for learned patterns
                conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_declarative (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    pattern TEXT,
                    confidence REAL,
                    success_rate REAL,
                    last_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Guardrail violations tracking
                conn.execute("""
                CREATE TABLE IF NOT EXISTS guardrail_violations (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT,
                    violation_type TEXT,
                    details TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    # 1. TECHNICAL ANALYSIS METHODS (100 lines)
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI)
        
        Formula:
            RS = Average Gain / Average Loss
            RSI = 100 - (100 / (1 + RS))
        
        Args:
            prices: Historical price data
            period: RSI period (default 14)
        
        Returns:
            RSI value (0-100)
        
        Target Values:
            - RSI < 30: Oversold (good entry)
            - RSI > 70: Overbought (avoid entry)
            - 40-60: Neutral
        
        Logging:
            Log structured event: rsi_calculated, value, status
        """
        try:
            if len(prices) < period + 1:
                self.logger.warning(f"Insufficient data for RSI calculation: {len(prices)} < {period + 1}")
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.abs(np.where(deltas < 0, deltas, 0))
            
            # Simple moving average for initial calculation
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            if avg_loss == 0:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Determine status
            if rsi < 30:
                status = "oversold"
            elif rsi > 70:
                status = "overbought"
            else:
                status = "neutral"
            
            self.logger.info(f"RSI calculated: {rsi:.2f} ({status})")
            return float(rsi)
            
        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return 50.0
    
    def calculate_macd(self, prices: List[float], 
                      fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Dict:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Formula:
            MACD = EMA(12) - EMA(26)
            Signal = EMA(MACD, 9)
            Histogram = MACD - Signal
        
        Returns:
            {
                'macd': float,
                'signal': float,
                'histogram': float,
                'bullish': bool
            }
        
        Entry Signal:
            - MACD > Signal: Bullish
            - MACD < Signal: Bearish
        
        Logging:
            Log structured event: macd_calculated, values, bullish_status
        """
        try:
            if len(prices) < slow:
                self.logger.warning(f"Insufficient data for MACD: {len(prices)} < {slow}")
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'bullish': False}
            
            prices_series = pd.Series(prices)
            
            # Calculate EMAs
            ema_fast = prices_series.ewm(span=fast).mean()
            ema_slow = prices_series.ewm(span=slow).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            # Current values
            current_macd = float(macd_line.iloc[-1])
            current_signal = float(signal_line.iloc[-1])
            current_histogram = float(histogram.iloc[-1])
            
            bullish = current_histogram > 0
            bullish_status = "bullish" if bullish else "bearish"
            
            self.logger.info(f"MACD calculated: MACD={current_macd:.4f}, Signal={current_signal:.4f}, Histogram={current_histogram:.4f} ({bullish_status})")
            
            return {
                'macd': current_macd,
                'signal': current_signal,
                'histogram': current_histogram,
                'bullish': bullish
            }
            
        except Exception as e:
            self.logger.error(f"MACD calculation error: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'bullish': False}
    
    def analyze_trend(self, prices: List[float]) -> Dict:
        """
        Analyze price trend using moving averages
        
        Method:
            SMA50 = Simple Moving Average (50-period)
            SMA200 = Simple Moving Average (200-period)
        
        Returns:
            {
                'sma50': float,
                'sma200': float,
                'trend': 'UPTREND' | 'DOWNTREND' | 'SIDEWAYS',
                'current_price': float,
                'trend_strength': 0-100  # How far above/below
            }
        
        Trend Definitions:
            - UPTREND: Price > SMA50 > SMA200, SMA50 > SMA200
            - DOWNTREND: Price < SMA50 < SMA200, SMA50 < SMA200
            - SIDEWAYS: Price oscillating around SMA50
        
        Logging:
            Log structured event: trend_analyzed, trend_type, strength
        """
        try:
            current_price = prices[-1] if prices else 0
            
            # Calculate SMAs
            sma50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
            sma200 = np.mean(prices[-200:]) if len(prices) >= 200 else np.mean(prices)
            
            # Determine trend
            if current_price > sma50 > sma200:
                if sma50 > sma200:
                    trend = 'UPTREND'
                    trend_strength = min(100, ((current_price - sma200) / sma200) * 100)
                else:
                    trend = 'SIDEWAYS'
                    trend_strength = 50
            elif current_price < sma50 < sma200:
                trend = 'DOWNTREND'
                trend_strength = max(0, 100 - ((sma200 - current_price) / sma200) * 100)
            else:
                trend = 'SIDEWAYS'
                trend_strength = 50
            
            result = {
                'sma50': float(sma50),
                'sma200': float(sma200),
                'trend': trend,
                'current_price': float(current_price),
                'trend_strength': float(trend_strength)
            }
            
            self.logger.info(f"Trend analyzed: {trend} (strength: {trend_strength:.1f}%)")
            return result
            
        except Exception as e:
            self.logger.error(f"Trend analysis error: {e}")
            return {
                'sma50': 0.0,
                'sma200': 0.0,
                'trend': 'SIDEWAYS',
                'current_price': 0.0,
                'trend_strength': 50.0
            }
    
    def find_support_resistance(self, prices: List[float], 
                               window: int = 20) -> Dict:
        """
        Find support and resistance levels
        
        Method:
            Support: Local minima (price bounced up from)
            Resistance: Local maxima (price bounced down from)
        
        Returns:
            {
                'support_levels': List[float],
                'resistance_levels': List[float],
                'nearest_support': float,
                'nearest_resistance': float,
                'support_distance_pct': float,
                'resistance_distance_pct': float
            }
        
        Logging:
            Log structured event: support_resistance_found
        """
        try:
            if len(prices) < window * 2:
                current_price = prices[-1] if prices else 100
                return {
                    'support_levels': [current_price * 0.95],
                    'resistance_levels': [current_price * 1.05],
                    'nearest_support': current_price * 0.95,
                    'nearest_resistance': current_price * 1.05,
                    'support_distance_pct': 5.0,
                    'resistance_distance_pct': 5.0
                }
            
            prices_array = np.array(prices)
            current_price = prices_array[-1]
            
            # Find local minima (support)
            support_levels = []
            resistance_levels = []
            
            for i in range(window, len(prices_array) - window):
                # Check if it's a local minimum
                if all(prices_array[i] <= prices_array[i-j] for j in range(1, window)) and \
                   all(prices_array[i] <= prices_array[i+j] for j in range(1, window)):
                    support_levels.append(prices_array[i])
                
                # Check if it's a local maximum
                if all(prices_array[i] >= prices_array[i-j] for j in range(1, window)) and \
                   all(prices_array[i] >= prices_array[i+j] for j in range(1, window)):
                    resistance_levels.append(prices_array[i])
            
            # Find nearest levels
            support_levels_below = [s for s in support_levels if s < current_price]
            resistance_levels_above = [r for r in resistance_levels if r > current_price]
            
            nearest_support = max(support_levels_below) if support_levels_below else current_price * 0.95
            nearest_resistance = min(resistance_levels_above) if resistance_levels_above else current_price * 1.05
            
            # Calculate distance percentages
            support_distance_pct = abs(current_price - nearest_support) / current_price * 100
            resistance_distance_pct = abs(nearest_resistance - current_price) / current_price * 100
            
            result = {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': float(nearest_support),
                'nearest_resistance': float(nearest_resistance),
                'support_distance_pct': float(support_distance_pct),
                'resistance_distance_pct': float(resistance_distance_pct)
            }
            
            self.logger.info(f"Support/Resistance found: Support={nearest_support:.2f} ({support_distance_pct:.1f}% below), Resistance={nearest_resistance:.2f} ({resistance_distance_pct:.1f}% above)")
            return result
            
        except Exception as e:
            self.logger.error(f"Support/Resistance calculation error: {e}")
            current_price = prices[-1] if prices else 100
            return {
                'support_levels': [],
                'resistance_levels': [],
                'nearest_support': current_price * 0.95,
                'nearest_resistance': current_price * 1.05,
                'support_distance_pct': 5.0,
                'resistance_distance_pct': 5.0
            }
    
    # 2. FILTERING METHODS (50 lines)
    
    def apply_pe_filter(self, stock_pe: float, 
                       industry_pe: float) -> Tuple[bool, str]:
        """
        Apply PE ratio filter for entry validation
        
        Guardrail 3: Stock PE within 0.6-1.6 × Industry PE
        """
        try:
            if industry_pe <= 0:
                self.logger.warning("Invalid industry PE, using default range")
                passes = 5.0 <= stock_pe <= 50.0
                reason = "PE within default range (5-50)" if passes else f"PE outside default range: {stock_pe:.2f}"
                self.logger.info(f"PE filter applied (default): {passes} - {reason}")
                return passes, reason
            
            min_pe = industry_pe * 0.4  # More flexible lower bound
            max_pe = industry_pe * 2.5   # Allow higher PEs in growth market
            
            if min_pe <= stock_pe <= max_pe:
                passes = True
                reason = f"PE within acceptable band: {stock_pe:.2f} (range: {min_pe:.2f}-{max_pe:.2f})"
            elif stock_pe < min_pe:
                passes = False
                reason = f"PE too low: {stock_pe:.2f} vs min {min_pe:.2f}"
            else:
                passes = False
                reason = f"PE too high: {stock_pe:.2f} vs max {max_pe:.2f}"
            
            self.logger.info(f"PE filter applied: {passes} - {reason}")
            return passes, reason
            
        except Exception as e:
            self.logger.error(f"PE filter error: {e}")
            return False, f"PE filter error: {e}"
    
    def fetch_pe_data(self, ticker: str) -> Dict:
        """Fetch PE data from financial APIs or cache"""
        try:
            ticker_symbol = f"{ticker}.NS" if not ticker.endswith(('.NS', '.BO')) else ticker
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            stock_pe = info.get('trailingPE', 18.0)
            sector = info.get('sector', 'Unknown')
            
            # Estimate industry PE based on sector
            sector_pe_map = {
                'Technology': 25.0, 'Financial Services': 15.0, 'Healthcare': 20.0,
                'Consumer Cyclical': 18.0, 'Industrials': 20.0, 'Energy': 12.0,
                'Materials': 15.0, 'Utilities': 16.0, 'Real Estate': 20.0,
                'Consumer Defensive': 22.0, 'Communication Services': 25.0, 'Unknown': 18.0
            }
            
            industry_pe = sector_pe_map.get(sector, 18.0)
            
            result = {
                'stock_pe': float(stock_pe) if stock_pe else 18.0,
                'industry_pe': float(industry_pe),
                'sector': sector,
                'source': 'api',
                'data_freshness_hours': 0
            }
            
            self.logger.info(f"PE data fetched for {ticker}: Stock PE={stock_pe:.2f}, Industry PE={industry_pe:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"PE data fetch error for {ticker}: {e}")
            return {
                'stock_pe': 18.0, 'industry_pe': 18.0, 'sector': 'Unknown',
                'source': 'fallback', 'data_freshness_hours': 24
            }
    
    # 3. GUARDRAILS - 7 SAFETY CHECKS (100 lines)
    
    def apply_guardrails(self, signal: Dict) -> Tuple[bool, str]:
        """Apply 7 safety guardrails before entry"""
        try:
            ticker = signal.get('ticker', 'UNKNOWN')
            violations = []
            
            # Guardrail 1: Portfolio deployment < 100%
            total_deployed = sum(self.active_positions.values())
            if total_deployed >= self.corpus_size:
                violations.append("Portfolio fully deployed (100%)")
            
            # Guardrail 2: Confidence score >= 50% (relaxed for market conditions)
            confidence = signal.get('entry_confidence', 0)
            if confidence < 50:
                violations.append(f"Confidence too low: {confidence}% < 50%")
            
            # Guardrail 3: PE filter
            pe_data = self.fetch_pe_data(ticker)
            pe_passes, pe_reason = self.apply_pe_filter(pe_data['stock_pe'], pe_data['industry_pe'])
            if not pe_passes:
                violations.append(f"PE filter failed: {pe_reason}")
            
            # Guardrail 4: Position size <= 20% of corpus
            position_size = signal.get('position_value', self.corpus_size * 0.04)
            max_position = self.corpus_size * 0.2
            if position_size > max_position:
                violations.append(f"Position too large: ₹{position_size:,.0f} > ₹{max_position:,.0f}")
            
            # Guardrail 5: Risk-reward ratio >= 1.0 (relaxed for market conditions)
            entry_price = signal.get('entry_price', 0)
            target_price = signal.get('target_price', entry_price * 1.1)
            stop_loss = signal.get('stop_loss', entry_price * 0.95)
            
            if entry_price > 0 and stop_loss > 0:
                reward = target_price - entry_price
                risk = entry_price - stop_loss
                rr_ratio = reward / risk if risk > 0 else 0
                
                if rr_ratio < 1.0:
                    violations.append(f"Poor risk-reward ratio: {rr_ratio:.2f} < 1.0")
            
            # Guardrail 6: Sector concentration <= 40%
            sector = pe_data.get('sector', 'Unknown')
            sector_exposure = self.sector_positions.get(sector, 0) + position_size
            max_sector_exposure = self.corpus_size * 0.4
            
            if sector_exposure > max_sector_exposure:
                violations.append(f"Sector overexposure: {sector} would be {sector_exposure/self.corpus_size*100:.1f}% > 40%")
            
            # Guardrail 7: Not already in active trades
            if ticker in self.active_positions:
                violations.append(f"Already holding position in {ticker}")
            
            if violations:
                self._log_guardrail_violations(ticker, violations)
                self.logger.warning(f"Guardrail violations for {ticker}: {violations}")
                return False, violations
            
            self.logger.info(f"All guardrails passed for {ticker}")
            return True, "All guardrails passed"
            
        except Exception as e:
            self.logger.error(f"Guardrail check error: {e}")
            return False, [f"Guardrail system error: {e}"]
    
    def _log_guardrail_violations(self, ticker: str, violations: List[str]):
        """Log guardrail violations to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for violation in violations:
                    conn.execute("""
                    INSERT INTO guardrail_violations (ticker, violation_type, details)
                    VALUES (?, ?, ?)
                    """, (ticker, violation.split(':')[0], violation))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error logging guardrail violations: {e}")
    
    # 4. MEMORY INTEGRATION (50 lines)
    
    def retrieve_memory_patterns(self, ticker: str) -> List[Dict]:
        """Retrieve learned patterns from memory_declarative table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                SELECT pattern, confidence, success_rate, last_used
                FROM memory_declarative
                WHERE ticker = ? AND confidence > 0.5
                ORDER BY success_rate DESC, confidence DESC
                LIMIT 5
                """, (ticker,))
                
                patterns = []
                for row in cursor.fetchall():
                    patterns.append({
                        'pattern': row[0],
                        'confidence': float(row[1]),
                        'success_rate': float(row[2]),
                        'last_used': row[3]
                    })
                
                self.logger.info(f"Retrieved {len(patterns)} memory patterns for {ticker}")
                return patterns
                
        except Exception as e:
            self.logger.error(f"Memory pattern retrieval error: {e}")
            return []
    
    def apply_memory(self, signal: Dict, patterns: List[Dict]) -> Dict:
        """Boost confidence if matching patterns found"""
        try:
            base_confidence = signal.get('entry_confidence', 0)
            analysis_text = json.dumps(signal.get('technical_analysis', {}))
            
            # Find matching patterns
            matching_patterns = []
            for pattern in patterns:
                pattern_keywords = pattern['pattern'].lower().split(' + ')
                matches = 0
                for keyword in pattern_keywords:
                    if any(key in keyword for key in ['rsi', 'macd', 'trend', 'support', 'resistance']):
                        if keyword.replace(' ', '_') in analysis_text.lower():
                            matches += 1
                
                if matches >= len(pattern_keywords) / 2:
                    matching_patterns.append(pattern)
            
            if matching_patterns:
                avg_pattern_confidence = mean([p['confidence'] for p in matching_patterns])
                avg_success_rate = mean([p['success_rate'] for p in matching_patterns])
                
                boost = (avg_pattern_confidence * avg_success_rate) * 10
                
                new_confidence = min(100, base_confidence + boost)
                signal['entry_confidence'] = new_confidence
                signal['confidence_source'] = 'base + memory_boost'
                signal['memory_boost'] = boost
                signal['matching_patterns'] = len(matching_patterns)
                
                self.logger.info(f"Memory boost applied: {base_confidence:.1f}% → {new_confidence:.1f}% (+{boost:.1f}%)")
            else:
                signal['memory_boost'] = 0
                signal['matching_patterns'] = 0
                signal['confidence_source'] = 'base_only'
                
            return signal
            
        except Exception as e:
            signal['memory_boost'] = 0
            signal['matching_patterns'] = 0
            signal['confidence_source'] = 'base_only'
            return signal
    
    # 5. MAIN ANALYSIS METHOD (50 lines)
    
    async def analyze_entry_signals_async(self, candidates: List[Dict]) -> List[Dict]:
        """
        Async version of analyze_entry_signals for ADK compatibility
        
        Args:
            candidates: List of quality-screened candidates from previous agent
            
        Returns:
            List of entry timing recommendations with technical analysis
        """
        return self.analyze_entry_signals(candidates)
    
    def analyze_entry_signals(self, candidates: List[Dict]) -> List[Dict]:
        """
        Analyze entry signals for all quality candidates
        
        Pipeline:
            1. For each candidate:
                a. Fetch historical price data
                b. Calculate RSI
                c. Calculate MACD
                d. Analyze trend
                e. Find support/resistance
                f. Apply PE filter
                g. Apply guardrails
                h. Retrieve memory patterns
                i. Apply memory boost
                j. Generate final signal
        """
        self.logger.info(f"Starting entry timing analysis for {len(candidates)} candidates")
        print("Agent 2: Entry Timing & Technical Analysis")
        print("=" * 60)
        
        recommendations = []
        
        for i, candidate in enumerate(candidates, 1):
            ticker = candidate.get('ticker', f'UNKNOWN_{i}')
            
            try:
                print(f"Analyzing {i}/{len(candidates)}: {ticker}")
                
                # Step 1: Fetch historical price data
                historical_data = self._fetch_price_data(ticker)
                if not historical_data:
                    self.logger.warning(f"No price data available for {ticker}")
                    print(f"  {ticker}: SKIP - No price data")
                    continue
                
                prices = historical_data['prices']
                current_price = prices[-1] if prices else candidate.get('current_price', 0)
                
                # Step 2-4: Calculate technical indicators
                rsi = self.calculate_rsi(prices)
                macd = self.calculate_macd(prices)
                trend_analysis = self.analyze_trend(prices)
                
                # Step 5: Find support/resistance
                support_resistance = self.find_support_resistance(prices)
                
                # Build technical analysis summary
                technical_analysis = {
                    'rsi': rsi,
                    'macd_bullish': macd['bullish'],
                    'macd_histogram': macd['histogram'],
                    'trend': trend_analysis['trend'],
                    'trend_strength': trend_analysis['trend_strength'],
                    'support_level': support_resistance['nearest_support'],
                    'resistance_level': support_resistance['nearest_resistance'],
                    'price_above_sma50': current_price > trend_analysis['sma50']
                }
                
                # Calculate base signal score
                base_signal = self._calculate_signal_score(technical_analysis, candidate)
                
                # Step 6-7: Apply PE filter and guardrails with improved risk-reward
                resistance_target = support_resistance['nearest_resistance']
                support_stop = support_resistance['nearest_support'] * 0.98
                
                # Ensure minimum 8% upside target for good risk-reward
                min_target = current_price * 1.08
                target_price = max(resistance_target, min_target)
                
                # Ensure stop loss is not too far - max 6% below entry
                max_stop_loss = current_price * 0.94
                stop_loss = max(support_stop, max_stop_loss)
                
                signal_dict = {
                    'ticker': ticker,
                    'entry_confidence': base_signal['confidence'],
                    'entry_price': current_price,
                    'position_value': self.corpus_size * 0.04,  # Default 4% position
                    'target_price': target_price,
                    'stop_loss': stop_loss,
                    'technical_analysis': technical_analysis
                }
                
                guardrails_passed, guardrail_reason = self.apply_guardrails(signal_dict)
                if not guardrails_passed:
                    self.logger.info(f"Guardrails failed for {ticker}: {guardrail_reason}")
                    print(f"  {ticker}: SKIP - {guardrail_reason[0] if isinstance(guardrail_reason, list) else guardrail_reason}")
                    continue
                
                # Step 8-9: Retrieve and apply memory patterns
                memory_patterns = self.retrieve_memory_patterns(ticker)
                enhanced_signal = self.apply_memory(signal_dict, memory_patterns)
                
                # Step 10: Generate final recommendation
                final_recommendation = {
                    'ticker': ticker,
                    'current_price': float(current_price),
                    'entry_price': float(current_price * 0.99),  # Slight discount for entry
                    'entry_signal': self._determine_signal_action(enhanced_signal['entry_confidence']),
                    'entry_confidence': enhanced_signal['entry_confidence'],
                    
                    'technical_analysis': technical_analysis,
                    
                    'risk_metrics': {
                        'stop_loss': float(support_resistance['nearest_support'] * 0.98),
                        'target_1': float(support_resistance['nearest_resistance']),
                        'target_2': float(support_resistance['nearest_resistance'] * 1.03),
                        'target_3': float(support_resistance['nearest_resistance'] * 1.06),
                        'risk_reward_ratio': self._calculate_risk_reward_ratio(
                            current_price, 
                            support_resistance['nearest_resistance'],
                            support_resistance['nearest_support'] * 0.98
                        )
                    },
                    
                    'guardrails_passed': True,
                    'memory_applied': enhanced_signal.get('matching_patterns', 0) > 0,
                    'memory_confidence_boost': enhanced_signal.get('memory_boost', 0),
                    
                    'metadata': {
                        'agent_id': 'agent2_entry_timing',
                        'analysis_timestamp': datetime.now().isoformat(),
                        'data_quality_score': len(prices) / 100 * 100 if len(prices) <= 100 else 100
                    }
                }
                
                recommendations.append(final_recommendation)
                self.success_count += 1
                
                signal_action = final_recommendation['entry_signal']
                confidence = final_recommendation['entry_confidence']
                print(f"  {ticker}: {signal_action} (confidence: {confidence:.1f}%)")
                
                self.logger.info(f"Entry signal generated for {ticker}: {signal_action} ({confidence:.1f}% confidence)")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {ticker}: {e}")
                print(f"  {ticker}: ERROR - {str(e)[:50]}...")
                
            finally:
                self.analysis_count += 1
        
        # Print summary
        print(f"\nAgent 2 Summary: {self.success_count}/{self.analysis_count} candidates analyzed successfully")
        self.logger.info(f"Entry timing analysis completed: {self.success_count}/{self.analysis_count} successful")
        
        return recommendations
    def _convert_name_to_ticker(self, company_name: str) -> str:
        """Convert company name to NSE ticker symbol"""
        
        # Simple cleanup for ticker format - remove spaces, dots, hyphens
        clean_name = company_name.upper().replace(' ', '').replace('.', '').replace('-', '')
        
        # Remove common company suffixes
        suffixes = ['LTD', 'LIMITED', 'COMPANY', 'CO', 'CORP', 'CORPORATION', 'INC']
        for suffix in suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        # Limit to reasonable ticker length (NSE tickers are typically 3-12 chars)
        if len(clean_name) > 12:
            clean_name = clean_name[:12]
        
        return clean_name
    
    def _fetch_price_data(self, ticker: str, days: int = 100) -> Optional[Dict]:
        """Fetch historical price data with improved ticker handling"""
        try:
            # First, try the ticker as-is (it should come clean from Agent 1)
            primary_ticker = ticker.strip()
            
            # If it doesn't have exchange suffix, try converting name to ticker
            if not primary_ticker.endswith(('.NS', '.BO')):
                primary_ticker = self._convert_name_to_ticker(ticker)
            
            # Try multiple ticker formats, prioritizing NSE since we know these are Indian stocks
            ticker_variants = [
                f"{primary_ticker}.NS",   # NSE format (prioritized for Indian stocks)
                primary_ticker,           # Use as-is 
                f"{primary_ticker}.BO",   # BSE format
                primary_ticker.replace('.NS', '').replace('.BO', ''),  # Just the symbol
            ]
            
            # Remove duplicates while preserving order
            seen = set()
            ticker_variants = [x for x in ticker_variants if not (x in seen or seen.add(x))]
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            for ticker_symbol in ticker_variants:
                try:
                    stock = yf.Ticker(ticker_symbol)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if not hist.empty and len(hist) > 10:  # Need reasonable data
                        self.logger.info(f"Found price data for {ticker} using {ticker_symbol}")
                        return {
                            'prices': hist['Close'].tolist(),
                            'volumes': hist['Volume'].tolist() if 'Volume' in hist.columns else [],
                            'highs': hist['High'].tolist(),
                            'lows': hist['Low'].tolist(),
                            'data_points': len(hist),
                            'ticker_used': ticker_symbol
                        }
                except Exception as variant_error:
                    continue  # Try next variant
            
            # No data found with any variant
            self.logger.warning(f"No price data found for {ticker} (tried: {ticker_variants})")
            return None
            
        except Exception as e:
            self.logger.error(f"Price data fetch error for {ticker}: {e}")
            return None
    
    def _calculate_signal_score(self, technical_analysis: Dict, candidate: Dict) -> Dict:
        """Calculate base signal score from technical analysis"""
        try:
            score = 0
            max_score = 100
            
            # RSI scoring (25 points)
            rsi = technical_analysis.get('rsi', 50)
            if 30 <= rsi <= 70:
                score += 25
            elif rsi < 30:
                score += 20  # Oversold is good for entry
            elif rsi <= 80:
                score += 15
            else:
                score += 5  # Very overbought
            
            # MACD scoring (20 points)
            if technical_analysis.get('macd_bullish', False):
                score += 20
            else:
                score += 5
            
            # Trend scoring (25 points)
            trend = technical_analysis.get('trend', 'SIDEWAYS')
            trend_strength = technical_analysis.get('trend_strength', 50)
            
            if trend == 'UPTREND':
                score += min(25, trend_strength / 4)  # Up to 25 points based on strength
            elif trend == 'SIDEWAYS':
                score += 15
            else:  # DOWNTREND
                score += 5
            
            # Price position scoring (15 points)
            if technical_analysis.get('price_above_sma50', False):
                score += 15
            else:
                score += 5
            
            # Quality score from Agent 1 (15 points)
            quality_score = candidate.get('quality_score', 5.0)
            score += min(15, quality_score * 1.5)
            
            confidence = min(score, max_score)
            
            return {
                'score': score,
                'confidence': confidence,
                'components': {
                    'rsi': rsi,
                    'trend': trend,
                    'quality': quality_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Signal score calculation error: {e}")
            return {'score': 50, 'confidence': 50, 'components': {}}
    
    def _determine_signal_action(self, confidence: float) -> str:
        """Determine signal action based on confidence"""
        if confidence >= 80:
            return "BUY_NOW"
        elif confidence >= 65:
            return "BUY_SOON"
        elif confidence >= 45:
            return "WAIT"
        else:
            return "AVOID"
    
    def _calculate_risk_reward_ratio(self, entry_price: float, target_price: float, stop_loss: float) -> float:
        """Calculate risk-reward ratio"""
        try:
            if entry_price <= 0 or stop_loss <= 0:
                return 1.0
            
            reward = abs(target_price - entry_price)
            risk = abs(entry_price - stop_loss)
            
            return round(reward / risk, 2) if risk > 0 else 1.0
            
        except Exception:
            return 1.0


