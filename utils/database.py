"""
Database utilities for the Market Timing Agents system
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, db_path: str = "market_timing_agents.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database and tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create tables for each agent
                self.create_tables(conn)
                logger.info(f"Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def create_tables(self, conn: sqlite3.Connection):
        """Create necessary tables"""
        cursor = conn.cursor()
        
        # Quality screening results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_screening (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                quality_score REAL,
                selected BOOLEAN,
                criteria_met TEXT
            )
        ''')
        
        # Entry timing results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entry_timing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                entry_signal TEXT,
                confidence REAL,
                technical_indicators TEXT
            )
        ''')
        
        # Exit management results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exit_management (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stock_name TEXT NOT NULL,
                entry_price REAL,
                stop_loss REAL,
                target_price REAL,
                risk_reward_ratio REAL
            )
        ''')
        
        # Trading sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT,
                results TEXT
            )
        ''')
        
        conn.commit()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a query and return results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def insert_record(self, table: str, data: Dict[str, Any]) -> bool:
        """Insert a record into the specified table"""
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data.keys()])
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(data.values()))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Insert failed for table {table}: {e}")
            return False