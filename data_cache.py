#!/usr/bin/env python3
"""
Data Cache Module for Market Timing Agents
Caches historical stock data to avoid repeated API calls during backtesting
"""

import os
import pickle
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from utils.logging_config import get_logger

class DataCache:
    """
    Handles caching and retrieval of historical stock data
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.logger = get_logger("DataCache")
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            self.logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _get_cache_filename(self, ticker: str, start_date: str, end_date: str) -> str:
        """Generate cache filename for given parameters"""
        # Sanitize ticker for filename
        clean_ticker = ticker.replace('.', '_').replace('-', '_')
        return f"{clean_ticker}_{start_date}_{end_date}.pkl"
    
    def _is_cache_valid(self, filepath: str, max_age_hours: int = 24) -> bool:
        """Check if cache file exists and is not too old"""
        if not os.path.exists(filepath):
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
        return file_age.total_seconds() < (max_age_hours * 3600)
    
    def get_historical_data(self, 
                          ticker: str, 
                          start_date: str, 
                          end_date: str,
                          force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get historical data from cache or fetch from API
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, ignore cache and fetch fresh data
            
        Returns:
            DataFrame with historical data or None if failed
        """
        cache_filename = self._get_cache_filename(ticker, start_date, end_date)
        cache_filepath = os.path.join(self.cache_dir, cache_filename)
        
        # Try to load from cache first (unless force refresh)
        if not force_refresh and self._is_cache_valid(cache_filepath):
            try:
                with open(cache_filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.logger.info(f"Loaded {ticker} from cache ({len(data)} rows)")
                    return data
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {ticker}: {e}")
        
        # Fetch fresh data from API
        try:
            self.logger.info(f"Fetching fresh data for {ticker} ({start_date} to {end_date})")
            
            # Try NSE format first
            nse_ticker = f"{ticker}.NS" if not ticker.endswith(('.NS', '.BO')) else ticker
            
            stock = yf.Ticker(nse_ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if hist.empty:
                self.logger.warning(f"No data found for {ticker}")
                return None
            
            # Add technical indicators
            hist['RSI'] = self._calculate_rsi(hist['Close'].values)
            hist['MACD'], hist['MACD_Signal'] = self._calculate_macd(hist['Close'].values)
            hist['SMA_50'] = hist['Close'].rolling(50).mean()
            hist['SMA_200'] = hist['Close'].rolling(200).mean()
            
            # Cache the data
            try:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump(hist, f)
                self.logger.info(f"Cached {ticker} data ({len(hist)} rows)")
            except Exception as e:
                self.logger.warning(f"Failed to cache data for {ticker}: {e}")
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker}: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        deltas = pd.Series(prices).diff()
        gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def clear_cache(self, ticker: str = None):
        """
        Clear cache files
        
        Args:
            ticker: If specified, clear cache only for this ticker. Otherwise clear all.
        """
        if ticker:
            # Clear cache for specific ticker
            pattern = ticker.replace('.', '_').replace('-', '_')
            files_removed = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(pattern) and filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    files_removed += 1
            
            self.logger.info(f"Cleared {files_removed} cache files for {ticker}")
        else:
            # Clear entire cache
            files_removed = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
                    files_removed += 1
            
            self.logger.info(f"Cleared entire cache ({files_removed} files)")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        info = {
            'total_files': len(cache_files),
            'cache_size_mb': 0,
            'tickers': set(),
            'files': []
        }
        
        total_size = 0
        for filename in cache_files:
            filepath = os.path.join(self.cache_dir, filename)
            file_size = os.path.getsize(filepath)
            total_size += file_size
            
            # Extract ticker from filename
            ticker_part = filename.split('_')[0]
            info['tickers'].add(ticker_part)
            
            info['files'].append({
                'filename': filename,
                'size_kb': round(file_size / 1024, 2),
                'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        info['cache_size_mb'] = round(total_size / (1024 * 1024), 2)
        info['tickers'] = list(info['tickers'])
        
        return info

def main():
    """Test the data cache functionality"""
    cache = DataCache()
    
    # Test with blue-chip stocks
    blue_chip_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
    
    print("🗄️  Testing Data Cache System")
    print("=" * 50)
    
    for ticker in blue_chip_stocks:
        print(f"\n📊 Fetching {ticker}...")
        data = cache.get_historical_data(ticker, "2023-01-01", "2024-12-31")
        if data is not None:
            print(f"   ✓ Success: {len(data)} days of data")
        else:
            print(f"   ✗ Failed to fetch data")
    
    # Show cache info
    cache_info = cache.get_cache_info()
    print(f"\n📁 Cache Summary:")
    print(f"   Files: {cache_info['total_files']}")
    print(f"   Size: {cache_info['cache_size_mb']} MB")
    print(f"   Tickers: {', '.join(cache_info['tickers'])}")

if __name__ == "__main__":
    main()