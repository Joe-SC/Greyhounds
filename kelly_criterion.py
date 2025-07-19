"""
Kelly Criterion Bet Sizing for Greyhound Racing

Uses TrueSkill ratings to estimate win probabilities and calculate optimal bet sizes
using the Kelly Criterion for bankroll management.
"""

import pandas as pd
import numpy as np
import trueskill
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
import math


class KellyCriterion:
    """Kelly Criterion calculator for greyhound racing bets"""
    
    def __init__(self, trueskill_system):
        """
        Initialize with a TrueSkill system
        
        Args:
            trueskill_system: GreyhoundTrueSkill instance
        """
        self.trueskill_system = trueskill_system
    
    def estimate_win_probabilities(self, dog_names: List[str]) -> Dict[str, float]:
        """
        Estimate win probability for each dog in a race using TrueSkill ratings
        
        For multi-dog races, we use the Bradley-Terry model which converts
        pairwise comparison probabilities into win probabilities for the field.
        
        Args:
            dog_names: List of dog names in the race
            
        Returns:
            Dictionary mapping dog_name -> win_probability
        """
        if len(dog_names) < 2:
            raise ValueError("Need at least 2 dogs for probability calculation")
        
        # Get ratings for all dogs
        ratings = {}
        for dog in dog_names:
            ratings[dog] = self.trueskill_system.get_rating(dog)
        
        # Calculate expected scores using TrueSkill's performance distribution
        # Each dog's performance is normally distributed: N(μ, σ² + β²)
        ts_env = trueskill.global_env()
        beta_squared = ts_env.beta ** 2
        
        expected_scores = {}
        for dog, rating in ratings.items():
            # Expected performance = skill level (μ)
            # Variance = uncertainty² + game variance
            expected_scores[dog] = rating.mu
        
        # Convert to win probabilities using softmax of expected scores
        # This approximates the probability that each dog has the highest performance
        scores = np.array(list(expected_scores.values()))
        
        # Apply softmax to convert scores to probabilities
        # Use temperature scaling based on average uncertainty
        avg_uncertainty = np.mean([r.sigma for r in ratings.values()])
        temperature = max(0.1, avg_uncertainty)  # Prevent division by zero
        
        exp_scores = np.exp(scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Create result dictionary
        result = {}
        for i, dog in enumerate(dog_names):
            result[dog] = probabilities[i]
        
        return result
    
    def calculate_kelly_fraction(self, win_probability: float, odds: float) -> float:
        """
        Calculate optimal bet fraction using Kelly Criterion
        
        Kelly fraction = (bp - q) / b
        Where:
        - b = odds - 1 (net odds)
        - p = win probability
        - q = lose probability = 1 - p
        
        Args:
            win_probability: Estimated probability of winning (0-1)
            odds: Decimal odds (e.g., 3.5 means 3.5:1)
            
        Returns:
            Optimal fraction of bankroll to bet (0-1)
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        if odds <= 1:
            return 0.0  # No positive expected value
        
        b = odds - 1  # Net odds
        p = win_probability
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Only bet if positive expected value
        return max(0.0, kelly_fraction)
    
    def calculate_expected_value(self, win_probability: float, odds: float, bet_amount: float) -> float:
        """
        Calculate expected value of a bet
        
        Args:
            win_probability: Probability of winning
            odds: Decimal odds
            bet_amount: Amount to bet
            
        Returns:
            Expected value of the bet
        """
        win_return = bet_amount * (odds - 1)  # Net profit if win
        lose_return = -bet_amount  # Loss if lose
        
        expected_value = (win_probability * win_return) + ((1 - win_probability) * lose_return)
        return expected_value
    
    def analyze_race_betting_opportunities(self, race_dogs: List[str], market_odds: Dict[str, float], 
                                         bankroll: float = 1000, max_fraction: float = 0.25) -> pd.DataFrame:
        """
        Analyze betting opportunities for a complete race
        
        Args:
            race_dogs: List of dog names in the race
            market_odds: Dictionary mapping dog_name -> decimal_odds
            bankroll: Total bankroll available
            max_fraction: Maximum fraction of bankroll to bet on any single dog
            
        Returns:
            DataFrame with betting analysis for each dog
        """
        # Get win probabilities
        win_probs = self.estimate_win_probabilities(race_dogs)
        
        results = []
        
        for dog in race_dogs:
            if dog not in market_odds:
                continue
                
            win_prob = win_probs[dog]
            odds = market_odds[dog]
            
            # Calculate Kelly fraction
            kelly_fraction = self.calculate_kelly_fraction(win_prob, odds)
            
            # Apply maximum bet constraint
            recommended_fraction = min(kelly_fraction, max_fraction)
            
            # Calculate bet amounts
            recommended_bet = bankroll * recommended_fraction
            
            # Calculate expected value
            expected_value = self.calculate_expected_value(win_prob, odds, recommended_bet)
            
            # Calculate value (difference between our probability and implied probability)
            implied_prob = 1 / odds
            value = win_prob - implied_prob
            
            # Get TrueSkill details
            rating = self.trueskill_system.get_rating(dog)
            
            results.append({
                'dog_name': dog,
                'skill': rating.mu,
                'uncertainty': rating.sigma,
                'conservative': rating.mu - 2 * rating.sigma,
                'estimated_win_prob': win_prob,
                'market_odds': odds,
                'implied_prob': implied_prob,
                'value': value,
                'kelly_fraction': kelly_fraction,
                'recommended_fraction': recommended_fraction,
                'recommended_bet': recommended_bet,
                'expected_value': expected_value,
                'positive_ev': expected_value > 0
            })
        
        df = pd.DataFrame(results)
        
        # Sort by expected value (best bets first)
        df = df.sort_values('expected_value', ascending=False)
        
        return df
    
    def fractional_kelly(self, kelly_fraction: float, fraction: float = 0.25) -> float:
        """
        Apply fractional Kelly to reduce risk
        
        Args:
            kelly_fraction: Full Kelly fraction
            fraction: Fraction of Kelly to use (e.g., 0.25 for quarter-Kelly)
            
        Returns:
            Adjusted Kelly fraction
        """
        return kelly_fraction * fraction
    
    def bankroll_simulation(self, bets: pd.DataFrame, initial_bankroll: float = 1000, 
                          num_simulations: int = 1000) -> Dict:
        """
        Simulate bankroll growth using the recommended bet sizes
        
        Args:
            bets: DataFrame from analyze_race_betting_opportunities
            initial_bankroll: Starting bankroll
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with simulation results
        """
        # Filter to positive EV bets only
        positive_bets = bets[bets['positive_ev'] == True].copy()
        
        if len(positive_bets) == 0:
            return {"message": "No positive expected value bets found"}
        
        final_bankrolls = []
        
        for _ in range(num_simulations):
            bankroll = initial_bankroll
            
            for _, bet in positive_bets.iterrows():
                if bankroll <= 0:
                    break
                
                bet_amount = bet['recommended_bet']
                win_prob = bet['estimated_win_prob']
                odds = bet['market_odds']
                
                # Simulate outcome
                if np.random.random() < win_prob:
                    # Win
                    bankroll += bet_amount * (odds - 1)
                else:
                    # Lose
                    bankroll -= bet_amount
            
            final_bankrolls.append(bankroll)
        
        final_bankrolls = np.array(final_bankrolls)
        
        return {
            'mean_final_bankroll': np.mean(final_bankrolls),
            'median_final_bankroll': np.median(final_bankrolls),
            'std_final_bankroll': np.std(final_bankrolls),
            'prob_profit': np.mean(final_bankrolls > initial_bankroll),
            'prob_loss_50pct': np.mean(final_bankrolls < initial_bankroll * 0.5),
            'percentile_5': np.percentile(final_bankrolls, 5),
            'percentile_95': np.percentile(final_bankrolls, 95),
            'max_bankroll': np.max(final_bankrolls),
            'min_bankroll': np.min(final_bankrolls)
        }


def format_betting_analysis(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the betting analysis DataFrame for display
    
    Args:
        analysis_df: Output from analyze_race_betting_opportunities
        
    Returns:
        Formatted DataFrame ready for display
    """
    display_df = analysis_df.copy()
    
    # Apply exact formatting to match dashboard pattern
    # TrueSkill metrics: 2 decimal places
    display_df['skill'] = display_df['skill'].astype(float).round(2)
    display_df['uncertainty'] = display_df['uncertainty'].astype(float).round(2)
    display_df['conservative'] = display_df['conservative'].astype(float).round(2)
    
    # Percentages: 1 decimal place (will add % in style.format)
    display_df['estimated_win_pct'] = (display_df['estimated_win_prob'] * 100).round(1)
    display_df['implied_pct'] = (display_df['implied_prob'] * 100).round(1)
    display_df['value_pct'] = (display_df['value'] * 100).round(1)
    display_df['kelly_pct'] = (display_df['kelly_fraction'] * 100).round(1)
    display_df['recommended_pct'] = (display_df['recommended_fraction'] * 100).round(1)
    
    # Money amounts: 2 decimal places
    display_df['bet_amount'] = display_df['recommended_bet'].round(2)
    display_df['expected_value'] = display_df['expected_value'].round(2)
    
    # Odds: 1 decimal place
    display_df['odds'] = display_df['market_odds'].round(1)
    
    # Select and rename columns for display
    display_cols = [
        'dog_name', 'skill', 'uncertainty', 'conservative',
        'estimated_win_pct', 'odds', 'implied_pct', 'value_pct', 
        'kelly_pct', 'recommended_pct', 'bet_amount', 'expected_value', 'positive_ev'
    ]
    
    display_df = display_df[display_cols].copy()
    display_df.columns = [
        'Dog Name', 'Skill (μ)', 'Uncertainty (σ)', 'Conservative (μ - 2σ)',
        'Est. Win %', 'Odds', 'Implied %', 'Value %', 
        'Kelly %', 'Rec. %', 'Bet Amount', 'Expected Value', 'Positive EV'
    ]
    
    return display_df


# Example usage
def example_usage():
    """Example of how to use the Kelly Criterion calculator"""
    
    # This would typically be loaded from your existing system
    from greyhound_trueskill import GreyhoundTrueSkill
    
    # Initialize TrueSkill system (this would have your actual ratings)
    ts_system = GreyhoundTrueSkill()
    
    # For demo purposes, add some fake ratings
    ts_system.ratings["Fast Dog"] = trueskill.Rating(mu=30, sigma=2)
    ts_system.ratings["Average Dog"] = trueskill.Rating(mu=25, sigma=3)
    ts_system.ratings["Slow Dog"] = trueskill.Rating(mu=20, sigma=4)
    
    # Initialize Kelly calculator
    kelly = KellyCriterion(ts_system)
    
    # Example race
    race_dogs = ["Fast Dog", "Average Dog", "Slow Dog"]
    market_odds = {
        "Fast Dog": 2.5,     # 2.5:1 odds
        "Average Dog": 3.0,   # 3.0:1 odds  
        "Slow Dog": 4.0      # 4.0:1 odds
    }
    
    # Analyze betting opportunities
    analysis = kelly.analyze_race_betting_opportunities(
        race_dogs=race_dogs,
        market_odds=market_odds,
        bankroll=1000,
        max_fraction=0.25
    )
    
    print("=== BETTING ANALYSIS ===")
    formatted = format_betting_analysis(analysis)
    print(formatted.to_string(index=False))
    
    # Run simulation
    print("\n=== BANKROLL SIMULATION ===")
    simulation = kelly.bankroll_simulation(analysis, initial_bankroll=1000)
    
    for key, value in simulation.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()