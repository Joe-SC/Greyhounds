"""
Betfair Historical Data Parser

Extracts greyhound race data from Betfair historical data files into pandas DataFrames.
"""

import pandas as pd
import json
import bz2
import os
from pathlib import Path
import glob
from datetime import datetime, date
from typing import Any
import warnings

class BetfairRaceParser:
    """Parser for Betfair historical greyhound racing data"""
    
    def __init__(self, data_path: str = "betfair_data/BASIC"):
        """
        Initialize parser
        
        Args:
            data_path: Path to the BASIC historical data directory
        """
        self.data_path = Path(data_path)
        
    def parse_race_file(self, file_path: str) -> dict[str, Any] | None:
        """
        Parse a single .bz2 file and extract race result if available
        
        Args:
            file_path: Path to the .bz2 file
            
        Returns:
            Dictionary with race data or None if no valid race found
        """
        try:
            with bz2.open(file_path, 'rt') as f:
                lines = f.readlines()
            
            # Parse JSON lines to find the final settlement
            race_data = None
            
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    
                    if 'mc' in data:
                        for market in data['mc']:
                            market_def = market.get('marketDefinition', {})
                            
                            # Only process WIN markets that are closed with results
                            if (market_def.get('marketType') == 'WIN' and 
                                market_def.get('status') == 'CLOSED' and
                                market_def.get('countryCode') == 'GB'):
                                
                                race_data = self._extract_race_data(market_def, market.get('id'))
                                
                except json.JSONDecodeError:
                    continue
            
            return race_data
            
        except Exception as e:
            # Skip files that can't be parsed
            return None
    
    def _extract_race_data(self, market_def: dict[str, Any], market_id: str) -> dict[str, Any]:
        """
        Extract structured race data from market definition
        
        Args:
            market_def: Market definition from Betfair data
            market_id: Market ID
            
        Returns:
            Dictionary with race information
        """
        # Extract basic race info
        race_info = {
            'market_id': market_id,
            'event_id': market_def.get('eventId'),
            'venue': market_def.get('venue'),
            'race_name': market_def.get('name'),
            'event_name': market_def.get('eventName'),
            'race_time': market_def.get('marketTime'),
            'settled_time': market_def.get('settledTime'),
            'country': market_def.get('countryCode'),
            'num_runners': len(market_def.get('runners', []))
        }
        
        # Extract runner results
        runners = []
        winner_found = False
        
        for runner in market_def.get('runners', []):
            # Clean dog name (remove trap number)
            dog_name = runner.get('name', '').strip()
            if '. ' in dog_name and dog_name[0].isdigit():
                dog_name = dog_name.split('. ', 1)[1]
            
            is_winner = runner.get('status') == 'WINNER'
            if is_winner:
                winner_found = True
            
            runner_data = {
                'runner_id': runner.get('id'),
                'dog_name': dog_name,
                'trap_number': runner.get('sortPriority'),
                'status': runner.get('status'),
                'is_winner': is_winner,
                'bsp': runner.get('bsp'),  # Betfair Starting Price
            }
            
            runners.append(runner_data)
        
        # Only return race if we have a clear winner
        if winner_found and len(runners) > 1:
            race_info['runners'] = runners
            return race_info
        
        return None
    
    def parse_date_range(self, 
                        start_date: str | date, 
                        end_date: str | date) -> pd.DataFrame:
        """
        Parse all race files within a date range
        
        Args:
            start_date: Start date (YYYY-MM-DD format or date object)
            end_date: End date (YYYY-MM-DD format or date object)
            
        Returns:
            DataFrame with race results
        """
        # Convert to pandas datetime for easy date range generation
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        print(f"Parsing races from {start_dt.date()} to {end_dt.date()}")
        
        # Generate all dates in range using pandas
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # Collect all files for the date range
        all_files = []
        
        for current_dt in date_range:
            year = current_dt.year
            month = current_dt.strftime('%b')  # Jan, Feb, etc.
            day = current_dt.day
            
            # Build path pattern for this date
            pattern = str(self.data_path / str(year) / month / str(day) / "*" / "*.bz2")
            files = glob.glob(pattern)
            all_files.extend(files)
        
        print(f"Found {len(all_files)} files to process across {len(date_range)} days")
        
        return self._process_files(all_files)
    
    def parse_all_races(self, year: int = 2025, months: list[str] | None = None) -> pd.DataFrame:
        """
        Parse all race files for specified year and months
        
        Args:
            year: Year to process (default 2025)
            months: List of month names to process (default: all available)
            
        Returns:
            DataFrame with race results
        """
        if months is None:
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']
        
        print(f"Parsing {year} data for months: {', '.join(months)}")
        
        # Find all .bz2 files for specified months
        all_files = []
        for month in months:
            pattern = str(self.data_path / str(year) / month / "*" / "*" / "*.bz2")
            files = glob.glob(pattern)
            all_files.extend(files)
        
        print(f"Found {len(all_files)} files to process")
        
        return self._process_files(all_files)
    
    def _process_files(self, file_list: list[str]) -> pd.DataFrame:
        """
        Process a list of files and return DataFrame
        
        Args:
            file_list: List of file paths to process
            
        Returns:
            DataFrame with race results
        """
        # Parse files and collect race data
        races = []
        processed_count = 0
        
        for file_path in file_list:
            race_data = self.parse_race_file(file_path)
            
            if race_data:
                races.append(race_data)
            
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count}/{len(file_list)} files, found {len(races)} races")
        
        print(f"Completed! Found {len(races)} valid races from {processed_count} files")
        
        if not races:
            print("No races found!")
            return pd.DataFrame()
        
        return self._create_dataframes(races)
    
    def _create_dataframes(self, races: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Convert race data to pandas DataFrames
        
        Args:
            races: List of race dictionaries
            
        Returns:
            DataFrame with race results (one row per dog per race)
        """
        # Flatten race data - one row per dog per race
        rows = []
        
        for race in races:
            race_info = {k: v for k, v in race.items() if k != 'runners'}
            
            for runner in race['runners']:
                row = {**race_info, **runner}
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Convert datetime columns
        if 'race_time' in df.columns:
            df['race_time'] = pd.to_datetime(df['race_time'])
        if 'settled_time' in df.columns:
            df['settled_time'] = pd.to_datetime(df['settled_time'])
        
        # Sort by race time
        df = df.sort_values('race_time').reset_index(drop=True)
        
        print(f"Created DataFrame with {len(df)} rows (dog-race combinations)")
        print(f"Covering {df['market_id'].nunique()} unique races")
        print(f"With {df['dog_name'].nunique()} unique dogs")
        print(f"At {df['venue'].nunique()} different venues")
        
        return df

def quick_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the race data"""
    print("\n=== RACE DATA SUMMARY ===")
    print(f"Total rows: {len(df):,}")
    print(f"Unique races: {df['market_id'].nunique():,}")
    print(f"Unique dogs: {df['dog_name'].nunique():,}")
    print(f"Date range: {df['race_time'].min()} to {df['race_time'].max()}")
    print(f"Venues: {df['venue'].nunique()}")
    
    print("\nTop venues by race count:")
    venue_counts = df.groupby('venue')['market_id'].nunique().sort_values(ascending=False)
    print(venue_counts.head(10))

# Example usage
if __name__ == "__main__":
    # Initialize parser
    parser = BetfairRaceParser()
    
    # Example 1: Parse specific date range
    df = parser.parse_date_range('2025-01-01', '2025-01-07')  # First week of January
    
    # Example 2: Parse all of January 2025
    # df = parser.parse_all_races(year=2025, months=['Jan'])
    
    # Show summary
    quick_summary(df)
    
    # Save to CSV
    df.to_csv('greyhound_races_sample.csv', index=False)
    print(f"\nSaved data to greyhound_races_sample.csv")