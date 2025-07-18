"""
Visualization utilities for the greyhound analytics dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class Visualizations:
    """Handles chart creation and visualization for the dashboard."""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_top_dogs_rating(self, leaderboard: pd.DataFrame, top_n: int = 20) -> go.Figure:
        """Create horizontal bar chart of top dogs by conservative rating."""
        if leaderboard.empty:
            return go.Figure()
            
        top_dogs = leaderboard.head(top_n)
        
        fig = go.Figure(go.Bar(
            x=top_dogs['conservative'],
            y=top_dogs['dog_name'],
            orientation='h',
            marker_color='skyblue',
            text=top_dogs['races'],
            texttemplate='%{text} races',
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Conservative Rating: %{x:.2f}<br>' +
                         'Skill: %{customdata[0]:.2f}<br>' +
                         'Uncertainty: %{customdata[1]:.2f}<br>' +
                         'Races: %{customdata[2]}<extra></extra>',
            customdata=np.column_stack((top_dogs['skill'], top_dogs['uncertainty'], top_dogs['races']))
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Greyhounds by Conservative Rating',
            xaxis_title='Conservative Rating (μ - 2σ)',
            yaxis_title='Dog Name',
            height=max(400, top_n * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_skill_vs_uncertainty(self, leaderboard: pd.DataFrame) -> go.Figure:
        """Create scatter plot of skill vs uncertainty colored by race count."""
        if leaderboard.empty:
            return go.Figure()
        
        # Ensure size values are positive by normalizing conservative ratings
        leaderboard_copy = leaderboard.copy()
        min_conservative = leaderboard_copy['conservative'].min()
        if min_conservative < 0:
            # Shift all values to be positive
            leaderboard_copy['size_normalized'] = leaderboard_copy['conservative'] - min_conservative + 1
        else:
            leaderboard_copy['size_normalized'] = leaderboard_copy['conservative']
        
        # Ensure minimum size for visibility
        leaderboard_copy['size_normalized'] = np.maximum(leaderboard_copy['size_normalized'], 1)
            
        fig = px.scatter(
            leaderboard_copy,
            x='skill',
            y='uncertainty',
            color='races',
            size='size_normalized',
            hover_data=['dog_name', 'conservative'],
            color_continuous_scale='viridis',
            title='Skill vs Uncertainty (Size = Conservative Rating, Color = Races)'
        )
        
        fig.update_layout(
            xaxis_title='Skill (μ)',
            yaxis_title='Uncertainty (σ)',
            height=600
        )
        
        return fig
    
    def plot_rating_distribution(self, leaderboard: pd.DataFrame) -> go.Figure:
        """Create histogram of skill rating distribution."""
        if leaderboard.empty:
            return go.Figure()
            
        fig = go.Figure(go.Histogram(
            x=leaderboard['skill'],
            nbinsx=25,
            marker_color='lightgreen',
            opacity=0.7
        ))
        
        mean_skill = leaderboard['skill'].mean()
        fig.add_vline(
            x=mean_skill,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_skill:.1f}"
        )
        
        fig.update_layout(
            title='Distribution of Skill Ratings',
            xaxis_title='Skill Rating (μ)',
            yaxis_title='Number of Dogs',
            height=400
        )
        
        return fig
    
    def plot_venue_performance(self, venue_stats: pd.DataFrame) -> go.Figure:
        """Create venue performance comparison chart."""
        if venue_stats.empty:
            return go.Figure()
            
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Winner Skill by Venue', 'Race Count by Venue'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Average winner skill
        fig.add_trace(
            go.Bar(
                x=venue_stats['venue'],
                y=venue_stats['avg_winner_skill'],
                name='Avg Winner Skill',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Race count
        fig.add_trace(
            go.Bar(
                x=venue_stats['venue'],
                y=venue_stats['races'],
                name='Race Count',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="Venue Performance Analysis"
        )
        
        return fig
    
    def plot_dog_activity(self, df: pd.DataFrame, min_races: int = 3) -> go.Figure:
        """Create visualization of dog activity and win rates."""
        if df.empty:
            return go.Figure()
            
        # Calculate dog activity stats
        dog_activity = df.groupby('dog_name').agg({
            'market_id': 'nunique',
            'is_winner': ['sum', 'mean']
        })
        dog_activity.columns = ['races', 'wins', 'win_rate']
        dog_activity = dog_activity[dog_activity['races'] >= min_races]
        
        if dog_activity.empty:
            return go.Figure()
        
        # Ensure size values are positive (wins should already be positive, but just in case)
        dog_activity['size_normalized'] = np.maximum(dog_activity['wins'], 1)
        
        fig = px.scatter(
            dog_activity.reset_index(),
            x='races',
            y='win_rate',
            size='size_normalized',
            hover_data=['dog_name', 'wins'],
            title=f'Dog Activity vs Win Rate (min {min_races} races)',
            labels={'races': 'Number of Races', 'win_rate': 'Win Rate'}
        )
        
        fig.update_layout(height=500)
        return fig
    
    def plot_dog_comparison_bars(self, comparison_df: pd.DataFrame) -> go.Figure:
        """Create bar chart comparing conservative ratings of multiple dogs."""
        if comparison_df.empty:
            return go.Figure()
        
        fig = go.Figure(go.Bar(
            x=comparison_df['dog_name'],
            y=comparison_df['conservative'],
            marker_color=['gold' if i == comparison_df['conservative'].idxmax() else 'skyblue' 
                         for i in comparison_df.index],
            text=comparison_df['conservative'].round(1),
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Conservative Rating Comparison',
            xaxis_title='Dog Name',
            yaxis_title='Conservative Rating (μ - 2σ)',
            height=400,
            xaxis_tickangle=45
        )
        
        return fig
    
    def plot_dog_comparison_radar(self, comparison_df: pd.DataFrame) -> go.Figure:
        """Create radar chart comparing multiple metrics for dogs."""
        if comparison_df.empty:
            return go.Figure()
        
        # Normalize metrics to 0-100 scale for radar chart
        metrics = ['skill', 'conservative', 'races']
        normalized_df = comparison_df.copy()
        
        for metric in metrics:
            min_val = normalized_df[metric].min()
            max_val = normalized_df[metric].max()
            if max_val > min_val:
                normalized_df[f'{metric}_norm'] = 100 * (normalized_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{metric}_norm'] = 100
        
        # Invert uncertainty (lower is better)
        min_unc = normalized_df['uncertainty'].min()
        max_unc = normalized_df['uncertainty'].max()
        if max_unc > min_unc:
            normalized_df['uncertainty_norm'] = 100 * (max_unc - normalized_df['uncertainty']) / (max_unc - min_unc)
        else:
            normalized_df['uncertainty_norm'] = 100
        
        fig = go.Figure()
        
        # Add trace for each dog
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (_, dog) in enumerate(normalized_df.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=[dog['skill_norm'], dog['uncertainty_norm'], dog['conservative_norm'], dog['races_norm']],
                theta=['Skill', 'Certainty', 'Conservative', 'Experience'],
                fill='toself',
                name=dog['dog_name'],
                line_color=colors[i % len(colors)],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Multi-Metric Comparison (Normalized to 0-100)",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def plot_time_series_races(self, df: pd.DataFrame) -> go.Figure:
        """Create time series plot of race activity."""
        if df.empty or 'race_time' not in df.columns:
            return go.Figure()
            
        # Group by date
        daily_races = df.groupby(df['race_time'].dt.date).agg({
            'market_id': 'nunique'
        }).reset_index()
        daily_races.columns = ['date', 'races']
        
        fig = px.line(
            daily_races,
            x='date',
            y='races',
            title='Daily Race Count Over Time'
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Races',
            height=400
        )
        
        return fig
    
    def plot_track_comparison(self, df: pd.DataFrame, leaderboard: pd.DataFrame) -> go.Figure:
        """Create track-by-track comparison of average ratings."""
        if df.empty or leaderboard.empty:
            return go.Figure()
            
        # Merge ratings with race data
        df_with_ratings = df.merge(
            leaderboard[['dog_name', 'skill', 'conservative']], 
            on='dog_name', 
            how='left'
        )
        
        # Calculate average skill by venue
        venue_avg = df_with_ratings.groupby('venue')['skill'].mean().sort_values(ascending=False)
        
        if venue_avg.empty:
            return go.Figure()
        
        fig = go.Figure(go.Bar(
            x=venue_avg.index,
            y=venue_avg.values,
            marker_color='gold'
        ))
        
        fig.update_layout(
            title='Average Dog Skill Rating by Venue',
            xaxis_title='Venue',
            yaxis_title='Average Skill Rating',
            xaxis_tickangle=45,
            height=500
        )
        
        return fig