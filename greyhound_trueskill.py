"""
Greyhound TrueSkill Rating System

Uses Microsoft's TrueSkill algorithm to rate greyhounds based on race results.
Perfect for winner-takes-all scenarios with multiple competitors.
"""

import pandas as pd
import numpy as np
import trueskill
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class GreyhoundTrueSkill:
    """TrueSkill rating system for greyhounds"""
    
    def __init__(self, 
                 mu: float = 25.0,         # Default skill level
                 sigma: float = 8.333,     # Uncertainty (25/3)
                 beta: float = 4.167,      # Distance that guarantees ~76% win probability
                 tau: float = 0.083,       # Dynamics factor (prevents stagnation)
                 draw_probability: float = 0.0):  # No draws in greyhound racing
        """
        Initialize TrueSkill environment
        
        Args:
            mu: Default mean skill level
            sigma: Default skill uncertainty  
            beta: Distance that guarantees ~76% win probability
            tau: Dynamics factor (how much skill can change)
            draw_probability: Probability of draws (0 for greyhound racing)
        """
        # Configure TrueSkill environment
        self.env = trueskill.TrueSkill(
            mu=mu,
            sigma=sigma, 
            beta=beta,
            tau=tau,
            draw_probability=draw_probability
        )
        trueskill.setup(env=self.env)
        
        # Store ratings for each dog
        self.ratings = defaultdict(lambda: trueskill.Rating())
        self.race_history = []
        
    def update_from_race(self, race_results: list[tuple[str, bool]]) -> None:
        """
        Update ratings based on a single race
        
        Args:
            race_results: List of (dog_name, is_winner) tuples
        """
        if len(race_results) < 2:
            return  # Need at least 2 dogs
        
        # Separate winners and losers
        winners = [dog for dog, won in race_results if won]
        losers = [dog for dog, won in race_results if not won]
        
        if not winners or not losers:
            return  # Need both winners and losers
        
        # Get current ratings
        winner_ratings = [self.ratings[dog] for dog in winners]
        loser_ratings = [self.ratings[dog] for dog in losers]
        
        # TrueSkill expects list of teams, where each team is a list of players
        # Winners beat losers, losers all tie with each other
        teams = [winner_ratings, loser_ratings]
        ranks = [0, 1]  # Winners rank 0 (1st), losers rank 1 (tied for 2nd)
        
        # Calculate new ratings
        new_ratings = trueskill.rate(teams, ranks)
        
        # Update stored ratings
        for i, dog in enumerate(winners):
            self.ratings[dog] = new_ratings[0][i]
        for i, dog in enumerate(losers):
            self.ratings[dog] = new_ratings[1][i]
    
    def get_rating(self, dog_name: str) -> trueskill.Rating:
        """Get TrueSkill rating for a dog"""
        return self.ratings[dog_name]
    
    def get_conservative_rating(self, dog_name: str) -> float:
        """Get conservative rating estimate (mu - 2*sigma)"""
        rating = self.ratings[dog_name]
        return rating.mu - 2 * rating.sigma
    
    def get_skill_estimate(self, dog_name: str) -> float:
        """Get skill estimate (just mu)"""
        return self.ratings[dog_name].mu
    
    def get_uncertainty(self, dog_name: str) -> float:
        """Get rating uncertainty (sigma)"""
        return self.ratings[dog_name].sigma
    
    def predict_win_probability(self, dog_a: str, dog_b: str) -> float:
        """Predict probability that dog_a beats dog_b in head-to-head"""
        rating_a = self.ratings[dog_a]
        rating_b = self.ratings[dog_b]
        
        delta_mu = rating_a.mu - rating_b.mu
        sum_sigma = rating_a.sigma ** 2 + rating_b.sigma ** 2
        ts = trueskill.global_env()
        
        denom = np.sqrt(2 * (ts.beta ** 2) + sum_sigma)
        return ts.cdf(delta_mu / denom)
    
    def get_leaderboard(self, min_races: int = 5, sort_by: str = 'conservative') -> pd.DataFrame:
        """
        Get dog leaderboard
        
        Args:
            min_races: Minimum races to be included
            sort_by: 'conservative', 'skill', or 'uncertainty'
        """
        # Count races per dog from history
        race_counts = defaultdict(int)
        for race in self.race_history:
            for dog_name, _ in race['results']:
                race_counts[dog_name] += 1
        
        # Build leaderboard data
        leaderboard_data = []
        for dog_name, rating in self.ratings.items():
            if race_counts[dog_name] >= min_races:
                leaderboard_data.append({
                    'dog_name': dog_name,
                    'skill': rating.mu,
                    'uncertainty': rating.sigma,
                    'conservative': rating.mu - 2 * rating.sigma,
                    'races': race_counts[dog_name]
                })
        
        df = pd.DataFrame(leaderboard_data)
        if len(df) == 0:
            return df
        
        # Sort by chosen metric
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df[['rank', 'dog_name', 'skill', 'uncertainty', 'conservative', 'races']]

class TrueSkillProcessor:
    """Process Betfair race data with TrueSkill ratings"""
    
    def __init__(self):
        self.trueskill_system = GreyhoundTrueSkill()
    
    def process_dataframe(self, df: pd.DataFrame) -> None:
        """
        Process a DataFrame of race results and update TrueSkill ratings
        
        Args:
            df: DataFrame with columns [market_id, dog_name, is_winner, race_time, venue, etc.]
        """
        print("Processing races with TrueSkill...")
        
        # Group by race and process each race
        races_processed = 0
        
        for market_id, race_df in df.groupby('market_id'):
            # Create race results format for TrueSkill
            race_results = [(row['dog_name'], row['is_winner']) 
                          for _, row in race_df.iterrows()]
            
            # Update TrueSkill ratings
            self.trueskill_system.update_from_race(race_results)
            
            # Store race history
            race_info = race_df.iloc[0]
            self.trueskill_system.race_history.append({
                'market_id': market_id,
                'date': race_info['race_time'],
                'venue': race_info['venue'],
                'race_name': race_info.get('race_name', ''),
                'results': race_results,
                'winner': next(name for name, won in race_results if won)
            })
            
            races_processed += 1
            if races_processed % 100 == 0:
                print(f"Processed {races_processed} races...")
        
        print(f"Completed! Processed {races_processed} races")
        print(f"Tracking {len(self.trueskill_system.ratings)} dogs")
    
    def get_leaderboard(self, **kwargs) -> pd.DataFrame:
        """Get current leaderboard"""
        return self.trueskill_system.get_leaderboard(**kwargs)
    
    def plot_rating_progression(self, dog_names: list[str], figsize: tuple[int, int] = (12, 8)):
        """Plot skill progression over time for specific dogs"""
        # This would require storing rating history over time
        # For now, just show current ratings
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Get leaderboard
        lb = self.get_leaderboard(min_races=1)
        top_dogs = lb.head(len(dog_names) if dog_names else 20)
        
        # Plot skill vs uncertainty
        axes[0].scatter(top_dogs['skill'], top_dogs['uncertainty'], alpha=0.7)
        axes[0].set_xlabel('Skill (μ)')
        axes[0].set_ylabel('Uncertainty (σ)')
        axes[0].set_title('Skill vs Uncertainty')
        
        # Plot conservative rating
        axes[1].barh(range(len(top_dogs)), top_dogs['conservative'])
        axes[1].set_yticks(range(len(top_dogs)))
        axes[1].set_yticklabels(top_dogs['dog_name'])
        axes[1].set_xlabel('Conservative Rating (μ - 2σ)')
        axes[1].set_title('Top Dogs by Conservative Rating')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_venue_performance(self) -> pd.DataFrame:
        """Analyze average performance by venue"""
        venue_data = []
        
        for race in self.trueskill_system.race_history:
            venue = race['venue']
            winner = race['winner']
            
            if winner in self.trueskill_system.ratings:
                rating = self.trueskill_system.ratings[winner]
                venue_data.append({
                    'venue': venue,
                    'winner_skill': rating.mu,
                    'winner_uncertainty': rating.sigma,
                    'date': race['date']
                })
        
        if not venue_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(venue_data)
        
        venue_stats = df.groupby('venue').agg({
            'winner_skill': ['mean', 'std', 'count'],
            'winner_uncertainty': 'mean'
        }).round(2)
        
        venue_stats.columns = ['avg_winner_skill', 'skill_std', 'races', 'avg_uncertainty']
        venue_stats = venue_stats.sort_values('avg_winner_skill', ascending=False)
        
        return venue_stats.reset_index()

# Example usage
def main():
    """Example of how to use the TrueSkill system"""
    from betfair_parser import BetfairRaceParser
    
    # Parse some race data
    parser = BetfairRaceParser()
    df = parser.parse_date_range('2025-01-01', '2025-01-03')  # 3 days of data
    
    if len(df) == 0:
        print("No race data found!")
        return
    
    # Process with TrueSkill
    processor = TrueSkillProcessor()
    processor.process_dataframe(df)
    
    # Show leaderboard
    print("\\n=== TOP GREYHOUNDS BY TRUESKILL ===")
    leaderboard = processor.get_leaderboard(min_races=2)
    print(leaderboard.head(20).to_string(index=False))
    
    # Show venue analysis
    print("\\n=== VENUE ANALYSIS ===")
    venue_stats = processor.analyze_venue_performance()
    print(venue_stats.to_string(index=False))
    
    # Save results
    leaderboard.to_csv('greyhound_trueskill_ratings.csv', index=False)
    print("\\nSaved TrueSkill ratings to greyhound_trueskill_ratings.csv")

if __name__ == "__main__":
    main()