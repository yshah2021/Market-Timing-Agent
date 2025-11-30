#!/usr/bin/env python3
"""
ADK Quality Screening Agent: Google ADK BaseAgent implementation
- Event-driven architecture with async generators
- State management via InvocationContext
- Structured logging and error handling
- KPI tracking and performance monitoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Official ADK imports
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

import pandas as pd
import yfinance as yf
import time
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
import asyncio

# Import system modules
from utils.logging_config import get_logger

class QualityScreeningAgent(BaseAgent):
    """
    ADK Quality Screening Agent: Quality Scorer & Risk Calculator
    
    Features:
    - Official ADK BaseAgent inheritance
    - Event-driven architecture with proper Event yielding
    - State management via InvocationContext  
    - Structured JSON logging
    - Error handling with graceful degradation
    - Performance monitoring
    - KPI tracking
    """
    
    model_config = {"arbitrary_types_allowed": True}
    name: str = "QualityScreeningAgent"
    description: str = "Analyzes stocks for fundamental quality and screening"
    logger: Any = None
    session_id: Any = None
    config: Any = None
    metrics: Any = None
    
    def __init__(self):
        """Initialize quality screening agent"""
        super().__init__(name="QualityScreeningAgent")  # Initialize BaseAgent first
        self.logger = get_logger('agent1')
        self.session_id = None
        
        # Configuration
        self.config = {
            'screener_url': 'https://www.screener.in/screens/3327459/star-stocks/',
            'max_candidates': 30,
            'quality_weights': {
                'roe': 0.25,
                'debt_equity': 0.20, 
                'growth': 0.20,
                'margin': 0.15,
                'peg': 0.20
            },
            'timeout_seconds': 30,
            'retry_attempts': 3
        }
        
        # Performance tracking
        self.metrics = {
            'stocks_screened': 0,
            'stocks_passed': 0,
            'errors_encountered': 0,
            'total_time_seconds': 0
        }
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Official ADK BaseAgent implementation: Quality screening pipeline
        
        Args:
            ctx: ADK InvocationContext containing session state and configuration
            
        Yields:
            Event: Progress updates and completion events with screened candidates
        """
        
        try:
            # Extract strategy configuration from context state
            user_strategy = ctx.session.state.get('strategy', {})
            tickers = ctx.session.state.get('tickers', [])
            
            # Yield start event with proper genai types
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Starting quality screening for {len(tickers)} tickers")]
                )
            )
            
            # Run screening pipeline
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="Fetching screener data from screener.in")]
                )
            )
            
            candidates = await self.run_screening(user_strategy)
            
            # Yield progress event
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Processed {len(candidates)} quality candidates")]
                )
            )
            
            # Store results in session state for next agent (official ADK pattern)
            ctx.session.state['quality_candidates'] = candidates
            ctx.session.state['screening_metrics'] = self.metrics
            
            # Yield completion event with structured results
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=
                        f"Quality screening complete: {len(candidates)} candidates selected with average quality score {sum(c.get('quality_score', 0) for c in candidates) / max(len(candidates), 1):.2f}"
                    )]
                )
            )
            
        except Exception as e:
            # Yield error event with proper error handling
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Quality screening failed: {str(e)}")]
                )
            )
    
    async def run_screening(self, user_strategy: dict) -> list:
        """
        Enhanced screening pipeline with enterprise features
        
        Args:
            user_strategy: User strategy configuration
            
        Returns:
            List of qualified candidates for Agent 2
        """
        
        method_name = f"{self.__class__.__name__}.run_screening"
        
        try:
            # Start session and performance tracking
            self.session_id = self.logger.start_session({
                'strategy': user_strategy,
                'max_candidates': self.config['max_candidates']
            })
            
            start_time = datetime.now()
            self.logger.start_timer(method_name)
            
            # Log screening start
            self.logger.info({
                "event": "screening_started",
                "session_id": self.session_id,
                "max_candidates": self.config['max_candidates']
            })
            
            # Step 1: Fetch screener data with retry logic
            screener_df = await self._fetch_screener_data_with_retry()
            
            if screener_df.empty:
                self.logger.warning({
                    "event": "screening_failed",
                    "reason": "no_screener_data",
                    "fallback_used": "none"
                })
                return []
            
            self.metrics['stocks_screened'] = len(screener_df)
            
            # Step 2: Process with quality scoring
            candidates = await self._process_screening_data(screener_df)
            
            self.metrics['stocks_passed'] = len(candidates)
            
            # Step 3: Calculate performance metrics
            duration = self.logger.end_timer(method_name)
            self.metrics['total_time_seconds'] = duration
            
            # Log screening completion
            self.logger.info({
                "event": "screening_completed",
                "session_id": self.session_id,
                "stocks_screened": self.metrics['stocks_screened'],
                "stocks_passed": self.metrics['stocks_passed'],
                "pass_rate_pct": (self.metrics['stocks_passed'] / max(self.metrics['stocks_screened'], 1)) * 100,
                "duration_seconds": duration
            })
            
            # Log system metrics
            self._log_performance_metrics()
            
            return candidates
            
        except Exception as e:
            # Log critical error
            self.logger.error({
                "event": "screening_critical_error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "session_id": self.session_id
            })
            
            # Return empty list on critical failure
            return []
    
    async def _fetch_screener_data_with_retry(self) -> pd.DataFrame:
        """Fetch screener data with retry logic and error handling"""
        
        method_name = f"{self.__class__.__name__}._fetch_screener_data_with_retry"
        
        for attempt in range(1, self.config['retry_attempts'] + 1):
            try:
                self.logger.info({
                    "event": "screener_fetch_attempt",
                    "attempt": attempt,
                    "url": self.config['screener_url']
                })
                
                # Fix SSL context for screener.in
                ssl._create_default_https_context = ssl._create_unverified_context
                
                all_stocks = []
                
                # Fetch multiple pages using pandas read_html
                for page in range(1, 4):  # Pages 1, 2, 3
                    try:
                        page_url = f"{self.config['screener_url']}?page={page}" if page > 1 else self.config['screener_url']
                        
                        self.logger.info({
                            "event": "page_fetch_started",
                            "page": page,
                            "url": page_url
                        })
                        
                        dfs = pd.read_html(page_url)
                        
                        if dfs and len(dfs) > 0:
                            page_table = max(dfs, key=len)
                            
                            if len(page_table) > 0:
                                # Add a simple ticker extraction based on common patterns
                                if 'Name' in page_table.columns:
                                    page_table['NSE_Ticker'] = page_table['Name'].apply(self._extract_nse_ticker_heuristic)
                                
                                all_stocks.append(page_table)
                                
                                self.logger.info({
                                    "event": "page_fetch_completed",
                                    "page": page,
                                    "stocks_found": len(page_table)
                                })
                            else:
                                break  # No more data
                        else:
                            break  # No tables found
                            
                    except Exception as page_error:
                        self.logger.warning({
                            "event": "page_fetch_failed",
                            "page": page,
                            "error_message": str(page_error)
                        })
                        
                        if page == 1:
                            raise page_error  # Re-raise if first page fails
                        else:
                            continue  # Skip failed pages after first
                
                if all_stocks:
                    # Combine all pages
                    main_table = pd.concat(all_stocks, ignore_index=True)
                    
                    # Clean and validate data
                    cleaned_df = self._clean_and_validate_data(main_table)
                    
                    # Log ticker mapping statistics
                    if 'NSE_Ticker' in cleaned_df.columns:
                        total_mapped = cleaned_df['NSE_Ticker'].notna().sum()
                        self.logger.info({
                            "event": "ticker_mapping_stats",
                            "total_stocks": len(cleaned_df),
                            "successfully_mapped": total_mapped,
                            "mapping_success_rate": f"{total_mapped/len(cleaned_df)*100:.1f}%"
                        })
                    
                    self.logger.info({
                        "event": "screener_fetch_success",
                        "total_stocks": len(cleaned_df),
                        "pages_fetched": len(all_stocks),
                        "attempt": attempt
                    })
                    
                    return cleaned_df
                
            except Exception as e:
                self.logger.warning({
                    "event": "screener_fetch_failed",
                    "attempt": attempt,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                
                if attempt == self.config['retry_attempts']:
                    # Log final failure
                    self.logger.error({
                        "event": "screener_fetch_all_attempts_failed",
                        "total_attempts": attempt,
                        "strategy": "return_empty"
                    })
                    
                    return pd.DataFrame()
                
                # Wait before retry
                await self._async_sleep(2 * attempt)  # Exponential backoff
        
        return pd.DataFrame()  # Empty DataFrame if all attempts fail
    
    def _extract_nse_ticker_heuristic(self, company_name: str) -> str:
        """Extract NSE ticker using heuristic mapping for common stocks"""
        
        # Common ticker mappings for major Indian stocks
        ticker_map = {
            'Maruti Suzuki': 'MARUTI',
            'Cummins India': 'CUMMINSIND', 
            'A B B': 'ABB',
            'Bharat Electron': 'BEL',
            'Mazagon Dock': 'MDL',
            'Shanthi Gears': 'SHANTIGEAR',
            'Ajax Engineering': 'AJANTPHARMA',  # Example
            'Force Motors': 'FORCEMOT',
            'Foseco India': 'FOSECOIND',
            'Stylam Industrie': 'STYLAMIND',
            'Tips Music': 'TIPSMUSIC',
            'Railtel Corpn.': 'RAILTEL',
            'Ratnamani Metals': 'RATNAMANI',
            'Transrail Light': 'TRANSRAIL',
            'Triveni Turbine': 'TRITURBINE',
            'Himadri Special': 'HIMADRI',
            'Nucleus Soft.': 'NUCLEUS',
            'Saksoft': 'SAKSOFT',
            'EIH Assoc.Hotels': 'EIHOTEL',
            'TajGVK Hotels': 'TAJGVK',
            'NESCO': 'NESCO',
            'Gravita India': 'GRAVITA',
            'Power Mech Proj.': 'POWERMECH',
            'L T Foods': 'LTFOODS',
            'Torrent Power': 'TORNTPOWER',
            'Godfrey Phillips': 'GODFRYPHLP',
            'Caplin Point Lab': 'CAPLIPOINT',
            'SML Mahindra': 'SMLISML',
            'Quality Power El': 'QUALPOWER',
            'Inventurus Knowl': 'INVENKRA',
            'HBL Engineering': 'HBLPOWER',
            'Cemindia Project': 'CEMINDIA',
            'Elecon Engg.Co': 'ELECON',
            'Banco Products': 'BANCOINDIA',
            'Apar Inds.': 'APARINDS',
            'BLS Internat.': 'BLSINDIA',
            'AGI Greenpac': 'AGIGREEN',
            'TCPL Packaging': 'TCPLPACK',
            'GNG Electronics': 'GNGELECT',
            'Schneider Elect.': 'SCHNEIDER',
            'Action Const.Eq.': 'ACE',
            'Ashoka Buildcon': 'ASHOKA',
            'Indrapr.Medical': 'INDRAPRASTHA',
            'Garden Reach Sh.': 'GRSE',
            'Sri Lotus': 'SRILOTUS',
            'Seshaasai Tech.': 'SESHAASAI',
            'MPS': 'MPSLTD',
            'Prudent Corp.': 'PRUDENT',
            'Ganesh Housing': 'GANESHHOUSING',
            'I R C T C': 'IRCTC',
            'Esab India': 'ESABINDIA',
            'Waaree Renewab.': 'WAAREE'
        }
        
        # Direct lookup first
        if company_name in ticker_map:
            return ticker_map[company_name]
        
        # Fallback: Clean company name for ticker format  
        clean_name = company_name.upper().replace(' ', '').replace('.', '').replace('-', '')
        
        # Remove common suffixes
        suffixes = ['LTD', 'LIMITED', 'COMPANY', 'CO', 'CORP', 'CORPORATION', 'INC']
        for suffix in suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        # Limit to reasonable ticker length
        if len(clean_name) > 12:
            clean_name = clean_name[:12]
        
        return clean_name
    
    async def _process_screening_data(self, df: pd.DataFrame) -> list:
        """Process screening data with error handling"""
        
        if df.empty:
            self.logger.warning({
                "event": "processing_skipped",
                "reason": "empty_dataframe"
            })
            return []
        
        try:
            self.logger.info({
                "event": "processing_started",
                "input_stocks": len(df)
            })
            
            # Compute quality scores for each stock
            processed_stocks = []
            
            for index, row in df.iterrows():
                try:
                    # Log individual stock evaluation
                    ticker = str(row.get('Name', f'Stock_{index}'))
                    
                    self.logger.log_structured(
                        event_type="stock_evaluation_started",
                        ticker=ticker,
                        metric_name="quality_score_calculation",
                        status="INFO"
                    )
                    
                    # Compute quality score
                    quality_score = self._compute_quality_score(row)
                    
                    # Compute risk helpers
                    risk_data = self._compute_risk_helpers(row)
                    
                    # Create candidate payload
                    candidate = self._to_agent2_payload(row, quality_score, risk_data)
                    processed_stocks.append(candidate)
                    
                    # Log successful evaluation
                    self.logger.log_structured(
                        event_type="stock_evaluated",
                        ticker=ticker,
                        metric_name="quality_score",
                        metric_value=quality_score,
                        status="PASS" if quality_score >= 5.0 else "FAIL",
                        details={"trend_strength": risk_data.get('trend_strength', 0)}
                    )
                    
                except Exception as stock_error:
                    ticker = str(row.get('Name', f'Stock_{index}'))
                    
                    self.logger.error({
                        "event": "stock_evaluation_failed",
                        "ticker": ticker,
                        "error_type": type(stock_error).__name__,
                        "error_message": str(stock_error)
                    })
                    
                    self.metrics['errors_encountered'] += 1
                    continue  # Skip this stock and continue
            
            # Sort by quality score and select top candidates
            processed_stocks.sort(key=lambda x: x['quality_score'], reverse=True)
            top_candidates = processed_stocks[:self.config['max_candidates']]
            
            self.logger.info({
                "event": "processing_completed",
                "processed_stocks": len(processed_stocks),
                "selected_candidates": len(top_candidates),
                "avg_quality_score": sum(c['quality_score'] for c in top_candidates) / len(top_candidates) if top_candidates else 0
            })
            
            return top_candidates
            
        except Exception as e:
            self.logger.error({
                "event": "processing_critical_error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return []
    
    def _compute_quality_score(self, row) -> float:
        """
        Compute quality score with error handling
        
        Quality Score Formula (0-10):
        ROE: 25%, D/E: 20%, Profit Growth: 20%, OPM: 15%, PEG: 20%
        """
        
        try:
            # Get values with safe conversion and defaults
            roe = self._safe_float_convert(row.get('ROE %', row.get('ROCE %', 15)))
            debt_eq = self._safe_float_convert(row.get('Debt / Eq', 0.5))
            growth = self._safe_float_convert(row.get('Profit Var 3Yrs %', 15))
            opm = self._safe_float_convert(row.get('OPM %', 15))
            peg = self._safe_float_convert(row.get('PEG Ratio', 1.0))
            
            # Calculate components (0-1 scale)
            q_roe = min(roe / 25, 1) if roe > 0 else 0
            q_de = max(0, min(1 - debt_eq / 1.0, 1))
            q_growth = min(growth / 25, 1) if growth > 0 else 0
            q_margin = min(opm / 20, 1) if opm > 0 else 0
            q_peg = min(2 / peg, 1) if peg > 0 else 0
            
            weights = self.config['quality_weights']
            score = 10 * (
                weights['roe'] * q_roe +
                weights['debt_equity'] * q_de +
                weights['growth'] * q_growth +
                weights['margin'] * q_margin +
                weights['peg'] * q_peg
            )
            
            return max(0, min(10, score))  # Clamp between 0-10
            
        except Exception as e:
            ticker = str(row.get('Name', 'Unknown'))
            
            self.logger.warning({
                "event": "quality_score_calculation_failed",
                "ticker": ticker,
                "error_message": str(e),
                "fallback_score": 5.0
            })
            
            return 5.0  # Default moderate score
    
    def _compute_risk_helpers(self, row) -> Dict[str, Any]:
        """Compute risk analysis helpers for Agent 2"""
        
        try:
            cmp = self._safe_float_convert(row.get('CMP Rs.', 0))
            sma_50 = self._safe_float_convert(row.get('50 DMA Rs.', cmp))
            sma_200 = self._safe_float_convert(row.get('200 DMA Rs.', cmp))
            
            # Support proxy (conservative estimate)
            support_proxy = min(cmp, sma_50) if sma_50 > 0 else cmp
            
            # Distance to support
            dist_to_support_pct = ((cmp - support_proxy) / cmp * 100) if cmp > 0 else 0
            
            # Trend strength (0-3 score)
            trend_strength = self._compute_trend_strength(cmp, sma_50, sma_200)
            
            # Trend label
            trend_labels = {0: "DOWNTREND", 1: "SIDEWAYS", 2: "MILD_UPTREND", 3: "STRONG_UPTREND"}
            trend_label = trend_labels.get(trend_strength, "UNKNOWN")
            
            return {
                'support_proxy': support_proxy,
                'dist_to_support_pct': dist_to_support_pct,
                'trend_strength': trend_strength,
                'trend_label': trend_label
            }
            
        except Exception as e:
            ticker = str(row.get('Name', 'Unknown'))
            
            self.logger.warning({
                "event": "risk_helpers_calculation_failed",
                "ticker": ticker,
                "error_message": str(e)
            })
            
            # Return safe defaults
            return {
                'support_proxy': self._safe_float_convert(row.get('CMP Rs.', 100)),
                'dist_to_support_pct': 0.0,
                'trend_strength': 1,
                'trend_label': "SIDEWAYS"
            }
    
    def _compute_trend_strength(self, cmp: float, sma_50: float, sma_200: float) -> int:
        """
        Compute trend strength score (0-3)
        +1 if CMP > 50 DMA
        +1 if 50 DMA > 200 DMA  
        +1 if 50 DMA / 200 DMA > 1.02
        """
        
        try:
            score = 0
            
            if cmp > sma_50:
                score += 1
            if sma_50 > sma_200:
                score += 1
            
            if sma_200 > 0:
                slope_ratio = sma_50 / sma_200
                if slope_ratio > 1.02:
                    score += 1
            
            return score
            
        except Exception:
            return 1  # Default neutral trend
    
    def _to_agent2_payload(self, row, quality_score: float, risk_data: Dict) -> Dict:
        """Format data for Agent 2 consumption with error handling"""
        
        try:
            # Use NSE ticker if available, otherwise fall back to company name
            nse_ticker = row.get('NSE_Ticker')
            company_name = str(row.get("Name", ""))
            
            # Prioritize NSE ticker, but ensure we have some identifier
            ticker_to_use = nse_ticker if pd.notna(nse_ticker) and nse_ticker else company_name
            
            return {
                "ticker": ticker_to_use,
                "company_name": company_name,
                "nse_ticker": nse_ticker if pd.notna(nse_ticker) else None,
                "current_price": self._safe_float_convert(row.get("CMP Rs.", 0)),
                "market_cap_cr": self._safe_float_convert(row.get("Mar Cap Rs.Cr.", 0)),
                "debt_to_equity": self._safe_float_convert(row.get("Debt / Eq", 0)),
                "profit_growth_3y": self._safe_float_convert(row.get("Profit Var 3Yrs %", 0)),
                "sales_growth_3y": self._safe_float_convert(row.get("Sales Var 3Yrs %", 0)),
                "roe": self._safe_float_convert(row.get("ROE %", 0)),
                "roce": self._safe_float_convert(row.get("ROCE %", 0)),
                "opm": self._safe_float_convert(row.get("OPM %", 0)),
                "peg_ratio": self._safe_float_convert(row.get("PEG Ratio", 1)),
                "promoter_holding": self._safe_float_convert(row.get("Prom. Hold. %", 0)),
                "promoter_holding_change": self._safe_float_convert(row.get("Change in Prom Hold %", 0)),
                "sma_50": self._safe_float_convert(row.get("50 DMA Rs.", 0)),
                "sma_200": self._safe_float_convert(row.get("200 DMA Rs.", 0)),
                "quality_score": quality_score,
                "support_proxy": risk_data['support_proxy'],
                "dist_to_support_pct": risk_data['dist_to_support_pct'],
                "trend_strength": risk_data['trend_strength'],
                "trend_label": risk_data['trend_label'],
                "ready_for_agent_2": True,
                "screening_timestamp": datetime.now().isoformat(),
                "has_nse_ticker": pd.notna(nse_ticker) and nse_ticker is not None
            }
            
        except Exception as e:
            ticker = str(row.get('Name', 'Unknown'))
            
            self.logger.error({
                "event": "payload_formatting_failed",
                "ticker": ticker,
                "error_message": str(e)
            })
            
            # Return minimal safe payload
            return {
                "ticker": ticker,
                "company_name": ticker,
                "current_price": 100.0,
                "quality_score": quality_score,
                "ready_for_agent_2": False,
                "has_nse_ticker": False,
                "error": "payload_formatting_failed"
            }
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate screener data"""
        
        try:
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Map common column variations
            column_mapping = {
                'CMP  Rs.': 'CMP Rs.',
                'Mar Cap  Rs.Cr.': 'Mar Cap Rs.Cr.',
                'ROCE  %': 'ROCE %',
                'Profit Var 3Yrs  %': 'Profit Var 3Yrs %',
                'Sales Var 3Yrs  %': 'Sales Var 3Yrs %'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df[new_name] = df[old_name]
            
            # Add missing columns with defaults
            required_columns = {
                'Name': 'Unknown Stock',
                'CMP Rs.': 100.0,
                'Mar Cap Rs.Cr.': 1000.0,
                'Debt / Eq': 0.5,
                'Profit Var 3Yrs %': 15.0,
                'Sales Var 3Yrs %': 15.0,
                'ROE %': 15.0,
                'ROCE %': 15.0,
                'OPM %': 15.0,
                'Prom. Hold. %': 50.0,
                'Change in Prom Hold %': 0.0,
                'PEG Ratio': 1.0,
                '50 DMA Rs.': 100.0,
                '200 DMA Rs.': 100.0
            }
            
            for col, default_value in required_columns.items():
                if col not in df.columns:
                    df[col] = default_value
                    
                    self.logger.info({
                        "event": "column_added",
                        "column": col,
                        "default_value": default_value
                    })
            
            # Convert numeric columns
            numeric_cols = [col for col in required_columns.keys() if col != 'Name']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', ''), 
                        errors='coerce'
                    ).fillna(required_columns[col])
            
            self.logger.info({
                "event": "data_cleaned",
                "rows": len(df),
                "columns": len(df.columns)
            })
            
            return df
            
        except Exception as e:
            self.logger.error({
                "event": "data_cleaning_failed",
                "error_message": str(e)
            })
            
            return df  # Return original if cleaning fails
    
    def _safe_float_convert(self, value, default: float = 0.0) -> float:
        """Safely convert value to float with default"""
        
        try:
            if pd.isna(value):
                return default
            
            # Handle string values
            if isinstance(value, str):
                # Remove commas and convert
                clean_value = value.replace(',', '').strip()
                return float(clean_value) if clean_value else default
            
            # Handle numeric values
            return float(value)
            
        except (ValueError, TypeError):
            return default
    

    
    def _log_performance_metrics(self):
        """Log performance metrics for KPI calculation"""
        
        # Log system metrics
        self.logger.log_metric(
            metric_name="stocks_screened_count",
            metric_value=self.metrics['stocks_screened'],
            metric_unit="count"
        )
        
        self.logger.log_metric(
            metric_name="stocks_passed_count", 
            metric_value=self.metrics['stocks_passed'],
            metric_unit="count"
        )
        
        self.logger.log_metric(
            metric_name="screening_duration",
            metric_value=self.metrics['total_time_seconds'],
            metric_unit="seconds"
        )
        
        self.logger.log_metric(
            metric_name="error_count",
            metric_value=self.metrics['errors_encountered'],
            metric_unit="count"
        )
        
        # Calculate and log pass rate
        pass_rate = (self.metrics['stocks_passed'] / max(self.metrics['stocks_screened'], 1)) * 100
        
        self.logger.log_metric(
            metric_name="pass_rate",
            metric_value=pass_rate,
            metric_unit="percentage"
        )
    
    async def _async_sleep(self, seconds: float):
        """Async sleep helper"""
        import asyncio
        await asyncio.sleep(seconds)


# Don't create global instance - will be instantiated by ADK system when needed

# Main entry points
async def run_screening(user_strategy: dict) -> list:
    """
    Quality screening entry point
    
    Args:
        user_strategy: User strategy configuration
        
    Returns:
        List of qualified candidates for Agent 2
    """
    agent = QualityScreeningAgent()
    return await agent.run_screening(user_strategy)

async def run_stock_screening() -> list:
    """Compatibility function for test files"""
    return await run_screening({})