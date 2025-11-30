#!/usr/bin/env python3
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Official ADK imports
from google.adk.agents import SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

# Import system components
from utils.logging_config import get_logger

# Import ADK agents
from agents.quality_screening_agent import QualityScreeningAgent
from agents.entry_timing_agent import EntryTimingAgent  
from agents.exit_management_agent import ExitManagementAgent


class MarketTimingSequentialAgent(SequentialAgent):
    """
    Official ADK Market Timing Sequential Agent
    Orchestrates the complete 3-agent pipeline using SequentialAgent pattern
    
    Agent Flow:
    1. QualityScreeningAgent: Fundamental analysis and stock screening
    2. EntryTimingAgent: Technical analysis and entry signal generation  
    3. ExitManagementAgent: Exit planning and risk management
    
    Features:
    - Official ADK SequentialAgent inheritance
    - Event-driven architecture with proper Event yielding
    - State management via InvocationContext with proper state keys
    - Sequential execution with data flow between agents
    - Comprehensive logging and error handling
    - Safety guardrails and circuit breaker protection
    """
    
    model_config = {"arbitrary_types_allowed": True}
    name: str = "MarketTimingPipeline"
    quality_agent: Any = None
    entry_agent: Any = None
    exit_agent: Any = None
    session_id: str = ""
    config: Dict = {}
    corpus_size: float = 1000000
    max_positions: int = 10
    logger: Any = None
    system_metrics: Dict = {}
    
    def __init__(self, config: Dict = None, corpus_size: float = 1000000, max_positions: int = 10):
        """
        Initialize Official Market Timing Sequential Agent
        
        Args:
            config: System configuration dictionary
            corpus_size: Portfolio size for position sizing
            max_positions: Maximum number of concurrent positions
        """
        
        # Call parent SequentialAgent constructor first
        super().__init__(name="MarketTimingPipeline")
        
        # Initialize the three agents as instance variables
        self.quality_agent = QualityScreeningAgent()
        self.entry_agent = EntryTimingAgent()
        self.exit_agent = ExitManagementAgent()
        
        # Initialize system components
        self.session_id = str(uuid.uuid4())
        self.config = config or self._get_default_config()
        self.corpus_size = corpus_size
        self.max_positions = max_positions
        
        # System logger
        self.logger = get_logger('adk_market_timing')
        
        # Performance tracking
        self.system_metrics = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'error_count': 0,
            'avg_session_time': 0
        }
        
        self.logger.info({
            "event": "adk_system_initialized",
            "session_id": self.session_id,
            "agents_count": 3  # QualityAgent, EntryAgent, ExitAgent
        })

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Official ADK SequentialAgent implementation
        
        Runs the three-agent pipeline sequentially:
        1. QualityScreeningAgent: Screens for quality stocks
        2. EntryTimingAgent: Analyzes entry timing
        3. ExitManagementAgent: Creates exit strategies
        
        Args:
            ctx: ADK InvocationContext with session state
            
        Yields:
            Event: Progress events from each agent in the pipeline
        """
        
        try:
            # Yield pipeline start event
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="Starting market timing pipeline: Quality → Entry → Exit")]
                )
            )
            
            # Run Quality Screening Agent
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="Executing quality screening agent...")]
                )
            )
            
            async for event in self.quality_agent._run_async_impl(ctx):
                yield event
            
            # Run Entry Timing Agent (if we have quality candidates)
            quality_candidates = ctx.session.state.get('quality_candidates', [])
            if quality_candidates:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part(text=f"Quality screening completed. Running entry timing analysis for {len(quality_candidates)} candidates...")]
                    )
                )
                
                async for event in self.entry_agent._run_async_impl(ctx):
                    yield event
                
                # Run Exit Management Agent (if we have entry candidates)
                entry_candidates = ctx.session.state.get('entry_candidates', [])
                if entry_candidates:
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text=f"Entry analysis completed. Generating exit strategies for {len(entry_candidates)} positions...")]
                        )
                    )
                    
                    async for event in self.exit_agent._run_async_impl(ctx):
                        yield event
                        
                    # Final summary event
                    exit_plans = ctx.session.state.get('exit_plans', [])
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text=f"Market timing pipeline completed: {len(quality_candidates)} screened → {len(entry_candidates)} entries → {len(exit_plans)} exit plans")]
                        )
                    )
                else:
                    yield Event(
                        author=self.name,
                        content=types.Content(
                            parts=[types.Part(text="No viable entry candidates found. Pipeline completed.")]
                        )
                    )
            else:
                yield Event(
                    author=self.name,
                    content=types.Content(
                        parts=[types.Part(text="No quality candidates found. Pipeline completed.")]
                    )
                )
                
        except Exception as e:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"Pipeline error: {str(e)}")]
                )
            )
    
    
    async def run_complete_analysis_adk(self, strategy_config: Dict = None) -> Dict:
        """
        Official ADK-compatible entry point for complete market timing analysis
        
        Args:
            strategy_config: Strategy configuration dictionary
            
        Returns:
            Complete analysis results with agent outputs
        """
        
        try:
            # Create InvocationContext with strategy configuration (official pattern)
            ctx = InvocationContext()
            
            # Initialize session state with official state keys
            ctx.session.state['strategy'] = strategy_config or {}
            ctx.session.state['tickers'] = []  # Will be populated by screener
            ctx.session.state['corpus_size'] = self.corpus_size
            ctx.session.state['max_positions'] = self.max_positions
            
            self.logger.info({
                "event": "adk_analysis_started",
                "session_id": self.session_id,
                "strategy": strategy_config or {}
            })
            
            # Execute the sequential agent pipeline using official pattern
            events = []
            async for event in self._run_async_impl(ctx):
                events.append(event)
                self.logger.info({
                    "event": "adk_event_received", 
                    "event_author": getattr(event, 'author', 'unknown'),
                    "session_id": self.session_id
                })
            
            # Extract final results from context using official state keys
            results = {
                'quality_candidates': ctx.session.state.get('quality_candidates', []),
                'entry_candidates': ctx.session.state.get('entry_candidates', []),
                'exit_plans': ctx.session.state.get('exit_plans', []),
                'screening_metrics': ctx.session.state.get('screening_metrics', {}),
                'entry_metrics': ctx.session.state.get('entry_metrics', {}),
                'exit_metrics': ctx.session.state.get('exit_metrics', {}),
                'events': [{'author': getattr(e, 'author', 'unknown'), 'content': getattr(e, 'content', {})} for e in events],
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            self.system_metrics['successful_sessions'] += 1
            self.logger.info({
                "event": "adk_analysis_completed",
                "session_id": self.session_id,
                "exit_strategies_count": len(results['exit_plans'])
            })
            return results
            
        except Exception as e:
            self.system_metrics['error_count'] += 1
            self.logger.error({
                "event": "adk_analysis_failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "session_id": self.session_id
            })
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'success': False,
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat()
            }
        finally:
            self.system_metrics['total_sessions'] += 1
    
    def _get_default_config(self) -> Dict:
        """Get default system configuration"""
        return {
            'max_candidates': 25,
            'min_quality_score': 5.0,
            'risk_tolerance': 'MODERATE',
            'enable_guardrails': True,
            'log_level': 'INFO'
        }
    
    def get_system_status(self) -> Dict:
        """Get current system status for monitoring"""
        return {
            'session_id': self.session_id,
            'system_metrics': self.system_metrics,
            'timestamp': datetime.now().isoformat(),
            'agents_status': {
                'quality_screening': 'active',
                'entry_timing': 'active', 
                'exit_management': 'active'
            },
            'success_rate': (
                self.system_metrics['successful_sessions'] / 
                max(self.system_metrics['total_sessions'], 1)
            ) * 100
        }


# Don't create global instance - will be instantiated by ADK system when needed

# Backward compatibility function  
async def run_complete_analysis(user_strategy: dict = None):
    """Backward compatibility function for existing integrations"""
    system = MarketTimingSequentialAgent()
    return await system.run_complete_analysis_adk(user_strategy)


async def test_agents_directly():
    """Test agents directly without full ADK framework"""
    print("Testing Market Timing Agents directly...")
    
    # Test Agent 1: Quality Screening
    print("\n1. Testing Quality Screening Agent...")
    quality_agent = QualityScreeningAgent()
    
    try:
        # Run quality screening with sample strategy
        user_strategy = {'max_candidates': 10, 'min_quality_score': 5.0}
        quality_candidates = await quality_agent.run_screening(user_strategy)
        print(f"   ✅ Quality screening: {len(quality_candidates)} candidates found")
        
        if quality_candidates:
            # Test Agent 2: Entry Timing
            print("\n2. Testing Entry Timing Agent...")
            entry_agent = EntryTimingAgent()
            entry_recommendations = await entry_agent.analyze_entry_signals_async(quality_candidates[:5])
            print(f"   ✅ Entry timing: {len(entry_recommendations)} recommendations generated")
            
            if entry_recommendations:
                # Test Agent 3: Exit Management
                print("\n3. Testing Exit Management Agent...")
                exit_agent = ExitManagementAgent()
                exit_plans = await exit_agent.generate_exit_strategies_async(entry_recommendations[:3])
                print(f"   ✅ Exit management: {len(exit_plans)} exit plans created")
                
                return {
                    'quality_candidates': quality_candidates,
                    'entry_recommendations': entry_recommendations,
                    'exit_plans': exit_plans,
                    'success': True
                }
            else:
                print("   ⚠️  No entry recommendations generated")
        else:
            print("   ⚠️  No quality candidates found")
            
        return {'success': True, 'quality_candidates': quality_candidates}
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Main execution function testing agents directly"""
        print("Starting Direct Agent Testing...")
        results = await test_agents_directly()
        
        if results.get('success'):
            print(f"\n✅ Agent testing completed successfully!")
            print(f"Results Summary:")
            print(f"   • Quality candidates: {len(results.get('quality_candidates', []))}")
            print(f"   • Entry recommendations: {len(results.get('entry_recommendations', []))}")
            print(f"   • Exit plans: {len(results.get('exit_plans', []))}")
            
        else:
            print(f"\n❌ Agent testing failed: {results.get('error', 'Unknown error')}")
    
    # Run the async main function
    asyncio.run(main())