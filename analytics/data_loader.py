"""
Data loading and processing utilities for the greyhound analytics dashboard.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import sys
import pickle
import hashlib
from datetime import datetime

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from betfair_parser import BetfairRaceParser
from greyhound_trueskill import TrueSkillProcessor

class DataLoader:
    """Handles data loading and caching for the dashboard."""
    
    def __init__(self):
        self.parser = BetfairRaceParser()
        self.processor = TrueSkillProcessor()
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    @st.cache_data
    def load_race_data(_self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load race data for the specified date range."""
        try:
            df = _self.parser.parse_date_range(start_date, end_date)
            return df
        except Exception as e:
            st.error(f"Error loading race data: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data 
    def load_existing_data(_self, file_path: str = None) -> pd.DataFrame:
        """Load existing race data from CSV file."""
        if file_path is None:
            # Try to find existing data files
            possible_files = [
                'race_data_20250101_20250718.csv',
                'processed_race_data.csv',
                'race_data.csv'
            ]
            
            for file in possible_files:
                if Path(file).exists():
                    file_path = file
                    break
            
            if file_path is None:
                return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            # Convert date columns
            if 'race_time' in df.columns:
                df['race_time'] = pd.to_datetime(df['race_time'])
            return df
        except Exception as e:
            st.error(f"Error loading data from {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def _get_cache_key(self, df: pd.DataFrame) -> str:
        """Generate cache key based on data hash."""
        # Create hash from data shape and sample of content
        data_info = f"{len(df)}_{df.columns.tolist()}_{df.head().to_string()}"
        return hashlib.md5(data_info.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> tuple:
        """Load processed data from cache."""
        cache_file = self.cache_dir / f"trueskill_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None, None
    
    def _save_to_cache(self, cache_key: str, leaderboard: pd.DataFrame, venue_stats: pd.DataFrame):
        """Save processed data to cache."""
        cache_file = self.cache_dir / f"trueskill_{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((leaderboard, venue_stats), f)
        except Exception as e:
            st.warning(f"Could not save to cache: {e}")
    
    @st.cache_data
    def process_trueskill_ratings(_self, df: pd.DataFrame) -> tuple:
        """Process TrueSkill ratings from race data."""
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Check cache first
        cache_key = _self._get_cache_key(df)
        cached_result = _self._load_from_cache(cache_key)
        
        if cached_result[0] is not None:
            st.success("📦 Loaded TrueSkill ratings from cache!")
            return cached_result
            
        try:
            with st.spinner("🧠 Processing TrueSkill ratings... This may take a few minutes."):
                _self.processor.process_dataframe(df)
                leaderboard = _self.processor.get_leaderboard(min_races=1)
                venue_stats = _self.processor.analyze_venue_performance()
                
                # Save to cache
                _self._save_to_cache(cache_key, leaderboard, venue_stats)
                st.success("✅ TrueSkill processing complete and cached!")
                
                return leaderboard, venue_stats
        except Exception as e:
            st.error(f"Error processing TrueSkill ratings: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    @st.cache_data
    def load_existing_ratings(_self, file_path: str = 'greyhound_trueskill_ratings.csv') -> pd.DataFrame:
        """Load existing TrueSkill ratings from CSV."""
        try:
            if Path(file_path).exists():
                return pd.read_csv(file_path)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading ratings from {file_path}: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_existing_venue_stats(_self, file_path: str = 'venue_performance_analysis.csv') -> pd.DataFrame:
        """Load existing venue performance stats from CSV."""
        try:
            if Path(file_path).exists():
                return pd.read_csv(file_path)
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading venue stats from {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics for the dataset."""
        if df.empty:
            return {}
            
        stats = {
            'total_races': df['market_id'].nunique(),
            'total_dogs': df['dog_name'].nunique(),
            'total_venues': df['venue'].nunique(),
            'date_range': {
                'start': df['race_time'].min() if 'race_time' in df.columns else None,
                'end': df['race_time'].max() if 'race_time' in df.columns else None
            },
            'avg_field_size': df.groupby('market_id').size().mean(),
            'total_rows': len(df)
        }
        
        return stats
    
    def get_trueskill_processor(self):
        """Get the TrueSkill processor instance."""
        return self.processor