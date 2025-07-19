"""
Streamlit dashboard for greyhound racing TrueSkill analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from analytics.data_loader import DataLoader
    from analytics.visualizations import Visualizations
    from kelly_criterion import KellyCriterion, format_betting_analysis
except ImportError:
    # Fallback for relative imports
    from data_loader import DataLoader
    from visualizations import Visualizations
    try:
        from kelly_criterion import KellyCriterion, format_betting_analysis
    except ImportError:
        KellyCriterion = None
        format_betting_analysis = None

class GreyhoundDashboard:
    """Main dashboard class for the Streamlit app."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.viz = Visualizations()
        
        # Set page config
        st.set_page_config(
            page_title="Greyhound TrueSkill Analytics",
            page_icon="🐕",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Main dashboard application."""
        st.title("🐕 Greyhound TrueSkill")
        st.markdown("""
Interactive analysis of greyhound racing performance using [TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) ratings.\n
Made by Joe to help Brontë gamble on her birthday.
        """)
        # Load data first to get venue options
        df, leaderboard, venue_stats = self.load_dashboard_data()
        
        # Render sidebar with venue options available
        self.render_sidebar(df)
        
        if df.empty:
            st.warning("No data available. Please check your data sources or date range.")
            return
        
        # Main dashboard content
        self.render_overview(df, leaderboard, venue_stats)
        
        # Show dog search results if there's a search query
        if st.session_state.get('search_dogs'):
            self.render_dog_search_results(df, leaderboard)
        
        # Show dog comparison if there are dogs to compare
        if st.session_state.get('compare_dogs'):
            self.render_dog_comparison(df, leaderboard)
        
        # Show betting tool if enabled
        if st.session_state.get('enable_betting') and KellyCriterion is not None:
            self.render_betting_tool(df, leaderboard)
        
        self.render_leaderboard_section(df, leaderboard)
        self.render_visualizations(df, leaderboard, venue_stats)
        self.render_detailed_analysis(df, leaderboard)
    
    def render_sidebar(self, df: pd.DataFrame):
        """Render the sidebar controls."""
        st.sidebar.header("📊 Dashboard Controls")
        
        # Data source selection
        st.sidebar.subheader("Data Source")
        data_source = st.sidebar.radio(
            "Select data source:",
            ["Load Existing Data", "Parse New Data"]
        )
        
        if data_source == "Parse New Data":
            st.sidebar.subheader("Date Range")
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime(2025, 1, 1),
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=datetime.now(),
                    key="end_date"
                )
            
            st.session_state.data_source = "new"
            st.session_state.start_date = start_date.strftime('%Y-%m-%d')
            st.session_state.end_date = end_date.strftime('%Y-%m-%d')
        else:
            st.session_state.data_source = "existing"
        
        # Dog search functionality
        st.sidebar.subheader("🐕 Dog Lookup")
        st.session_state.search_dogs = st.sidebar.text_input(
            "Search for specific dogs:",
            placeholder="Type dog name...",
            help="Enter part of a dog name to search for specific dogs"
        )
        
        # Multi-dog comparison
        st.sidebar.subheader("⚖️ Compare Dogs")
        st.session_state.compare_dogs = st.sidebar.text_area(
            "Compare multiple dogs:",
            placeholder="Enter dog names, one per line:\nAcomb Felix\nSwift Hostile\nProper Heiress",
            help="Enter dog names (one per line) to compare their ratings and stats",
            height=100
        )
        
        # Kelly Criterion betting tool
        if KellyCriterion is not None:
            st.sidebar.subheader("💰 Betting Tool (Kelly Criterion)")
            st.session_state.enable_betting = st.sidebar.checkbox(
                "Enable betting analysis",
                help="Use Kelly Criterion to calculate optimal bet sizes"
            )
            
            if st.session_state.get('enable_betting'):
                st.session_state.bankroll = st.sidebar.number_input(
                    "Bankroll (£):",
                    min_value=1.0,
                    value=1000.0,
                    step=50.0,
                    help="Your total betting bankroll"
                )
                
                st.session_state.max_bet_fraction = st.sidebar.slider(
                    "Max bet % of bankroll:",
                    min_value=1.0,
                    max_value=50.0,
                    value=25.0,
                    step=1.0,
                    help="Maximum percentage of bankroll to bet on any single race"
                ) / 100
        
        # Filtering options
        st.sidebar.subheader("🔍 Display Filters")
        
        # Store filter values in session state
        st.session_state.min_races = st.sidebar.slider(
            "Minimum races for leaderboard",
            min_value=1,
            max_value=20,
            value=2
        )
        
        st.session_state.top_n_dogs = st.sidebar.slider(
            "Top N dogs to display",
            min_value=10,
            max_value=50,
            value=20
        )
        
        # Venue filter (dynamically populated from loaded data)
        if not df.empty and 'venue' in df.columns:
            venue_options = ["All Venues"] + sorted(df['venue'].unique().tolist())
        else:
            venue_options = ["All Venues"]
            
        st.session_state.venue_filter = st.sidebar.selectbox(
            "Filter leaderboard by venue:",
            options=venue_options,
            help="Show only dogs that have raced at the selected venue"
        )
    
    def load_dashboard_data(self):
        """Load data based on user selections."""
        with st.spinner("Loading data..."):
            if st.session_state.get('data_source') == 'new':
                # Parse new data
                df = self.data_loader.load_race_data(
                    st.session_state.start_date,
                    st.session_state.end_date
                )
                
                if not df.empty:
                    leaderboard, venue_stats = self.data_loader.process_trueskill_ratings(df)
                else:
                    leaderboard, venue_stats = pd.DataFrame(), pd.DataFrame()
            else:
                # Load existing data
                df = self.data_loader.load_existing_data()
                leaderboard = self.data_loader.load_existing_ratings()
                venue_stats = self.data_loader.load_existing_venue_stats()
                
                # If no existing ratings, process them
                if leaderboard.empty and not df.empty:
                    leaderboard, venue_stats = self.data_loader.process_trueskill_ratings(df)
        
        return df, leaderboard, venue_stats
    
    def render_overview(self, df: pd.DataFrame, leaderboard: pd.DataFrame, venue_stats: pd.DataFrame):
        """Render the overview section with key metrics."""
        st.header("📈 Overview")
        
        # Calculate summary stats
        stats = self.data_loader.get_summary_stats(df)
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Races", f"{stats['total_races']:,}")
            with col2:
                st.metric("Unique Dogs", f"{stats['total_dogs']:,}")
            with col3:
                st.metric("Venues", stats['total_venues'])
            with col4:
                st.metric("Avg Field Size", f"{stats['avg_field_size']:.1f}")
            
            # Date range
            if stats['date_range']['start'] and stats['date_range']['end']:
                st.info(f"📅 Data covers: {stats['date_range']['start'].strftime('%Y-%m-%d')} to {stats['date_range']['end'].strftime('%Y-%m-%d')}")
    
    def render_dog_search_results(self, df: pd.DataFrame, leaderboard: pd.DataFrame):
        """Render search results for specific dogs."""
        search_query = st.session_state.get('search_dogs', '').strip()
        
        if not search_query:
            return
        
        st.header("🔍 Dog Search Results")
        
        # Search in leaderboard (case insensitive)
        search_results = leaderboard[
            leaderboard['dog_name'].str.contains(search_query, case=False, na=False)
        ].copy()
        
        if search_results.empty:
            st.warning(f"No dogs found matching '{search_query}'")
            
            # Suggest similar names
            all_dogs = leaderboard['dog_name'].tolist()
            suggestions = [dog for dog in all_dogs if search_query.lower() in dog.lower()][:5]
            if suggestions:
                st.info("💡 Did you mean one of these?")
                for suggestion in suggestions:
                    if st.button(f"🐕 {suggestion}", key=f"suggest_{suggestion}"):
                        st.session_state.search_dogs = suggestion
                        st.rerun()
            return
        
        # Display search results
        st.success(f"Found {len(search_results)} dog(s) matching '{search_query}'")
        
        for idx, dog in search_results.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.subheader(f"🐕 {dog['dog_name']}")
                    st.write(f"**Rank:** #{dog['rank']}")
                
                with col2:
                    st.metric("Skill", f"{dog['skill']:.1f}")
                
                with col3:
                    st.metric("Conservative", f"{dog['conservative']:.1f}")
                
                with col4:
                    st.metric("Races", dog['races'])
                
                # Additional details
                with st.expander(f"📊 Details for {dog['dog_name']}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Uncertainty (σ):** {dog['uncertainty']:.2f}")
                        
                        # Get recent races for this dog
                        dog_races = df[df['dog_name'] == dog['dog_name']].copy()
                        if not dog_races.empty:
                            wins = dog_races['is_winner'].sum()
                            win_rate = wins / len(dog_races)
                            st.write(f"**Win Rate:** {win_rate:.1%} ({wins}/{len(dog_races)})")
                            
                            # Venues this dog has raced at
                            venues = dog_races['venue'].unique()
                            st.write(f"**Venues:** {', '.join(venues[:3])}")
                            if len(venues) > 3:
                                st.write(f"... and {len(venues) - 3} more")
                    
                    with col_b:
                        # Recent performance if date data available
                        if 'race_time' in dog_races.columns:
                            dog_races['race_time'] = pd.to_datetime(dog_races['race_time'])
                            recent_races = dog_races.nlargest(5, 'race_time')
                            
                            st.write("**Recent Races:**")
                            for _, race in recent_races.iterrows():
                                result = "🥇 Won" if race['is_winner'] else f"📍 T{race.get('trap_number', '?')}"
                                date = race['race_time'].strftime('%d/%m')  # British format: day/month
                                venue = race['venue'][:8]  # Truncate venue name
                                st.write(f"• {date} - {venue} - {result}")
                
                st.divider()
    
    def render_dog_comparison(self, df: pd.DataFrame, leaderboard: pd.DataFrame):
        """Render comparison between multiple dogs."""
        compare_input = st.session_state.get('compare_dogs', '').strip()
        
        if not compare_input:
            return
        
        # Parse dog names from input
        dog_names = [name.strip() for name in compare_input.split('\n') if name.strip()]
        
        if len(dog_names) < 2:
            st.warning("Please enter at least 2 dog names to compare (one per line)")
            return
        
        st.header("⚖️ Dog Comparison")
        
        # Find matching dogs
        comparison_data = []
        not_found = []
        
        for dog_name in dog_names:
            # Try exact match first, then partial match
            exact_match = leaderboard[leaderboard['dog_name'].str.lower() == dog_name.lower()]
            if not exact_match.empty:
                comparison_data.append(exact_match.iloc[0])
            else:
                partial_match = leaderboard[leaderboard['dog_name'].str.contains(dog_name, case=False, na=False)]
                if not partial_match.empty:
                    comparison_data.append(partial_match.iloc[0])
                else:
                    not_found.append(dog_name)
        
        if not_found:
            st.warning(f"Could not find: {', '.join(not_found)}")
        
        if len(comparison_data) < 2:
            st.error("Need at least 2 valid dog names to compare")
            return
        
        comparison_df = pd.DataFrame(comparison_data)
        
        st.success(f"Comparing {len(comparison_df)} dogs")
        
        # Comparison table
        st.subheader("📊 Rating Comparison")
        
        # Calculate wins, win rates, percentiles, and last win for each dog
        comparison_df_enhanced = comparison_df.copy()
        total_dogs = len(leaderboard)
        
        for idx, dog in comparison_df_enhanced.iterrows():
            dog_races = df[df['dog_name'] == dog['dog_name']]
            if not dog_races.empty:
                wins = dog_races['is_winner'].sum()
                win_rate = wins / len(dog_races)
                comparison_df_enhanced.loc[idx, 'wins'] = wins
                comparison_df_enhanced.loc[idx, 'win_rate'] = win_rate
                
                # Calculate percentile (what % of dogs this dog is better than)
                rank = dog['rank']
                percentile = ((total_dogs - rank + 1) / total_dogs) * 100
                comparison_df_enhanced.loc[idx, 'percentile'] = percentile
                
                # Find last win
                if 'race_time' in dog_races.columns:
                    dog_races['race_time'] = pd.to_datetime(dog_races['race_time'])
                    wins_only = dog_races[dog_races['is_winner'] == True]
                    if not wins_only.empty:
                        last_win = wins_only['race_time'].max()
                        comparison_df_enhanced.loc[idx, 'last_win'] = last_win.strftime('%d/%m/%Y')
                    else:
                        comparison_df_enhanced.loc[idx, 'last_win'] = 'Never'
                else:
                    comparison_df_enhanced.loc[idx, 'last_win'] = 'Unknown'
            else:
                comparison_df_enhanced.loc[idx, 'wins'] = 0
                comparison_df_enhanced.loc[idx, 'win_rate'] = 0
                comparison_df_enhanced.loc[idx, 'percentile'] = 0
                comparison_df_enhanced.loc[idx, 'last_win'] = 'Never'
        
        # Prepare display data
        display_cols = ['dog_name', 'rank', 'percentile', 'skill', 'uncertainty', 'conservative', 'races', 'wins', 'win_rate', 'last_win']
        display_df = comparison_df_enhanced[display_cols].copy()
        
        # Apply exact formatting for each column
        display_df['skill'] = display_df['skill'].astype(float).round(2)
        display_df['uncertainty'] = display_df['uncertainty'].astype(float).round(2) 
        display_df['conservative'] = display_df['conservative'].astype(float).round(2)
        display_df['percentile'] = display_df['percentile'].astype(float).round(1)
        display_df['wins'] = display_df['wins'].astype(int)
        display_df['win_rate'] = (display_df['win_rate'].astype(float) * 100).round(1)
        
        display_df.columns = ['Dog Name', 'Rank', 'Top %', 'Skill (μ)', 'Uncertainty (σ)', 'Conservative (μ - 2σ)', 'Races', 'Wins', 'Win Rate (%)', 'Last Win']
        
        # Highlight best values with better dark mode support
        def highlight_best(s):
            if s.name in ['Skill (μ)', 'Conservative (μ - 2σ)', 'Races', 'Wins', 'Win Rate (%)', 'Top %']:
                max_val = s.max()
                return ['background-color: rgba(76, 175, 80, 0.3); font-weight: bold' if v == max_val else '' for v in s]
            elif s.name == 'Rank':
                min_val = s.min()
                return ['background-color: rgba(76, 175, 80, 0.3); font-weight: bold' if v == min_val else '' for v in s]
            elif s.name == 'Uncertainty (σ)':
                min_val = s.min()
                return ['background-color: rgba(76, 175, 80, 0.3); font-weight: bold' if v == min_val else '' for v in s]
            else:
                return [''] * len(s)
        
        # Apply styling with format preservation
        styled_df = display_df.style.apply(highlight_best, axis=0).format({
            'Skill (μ)': '{:.2f}',
            'Uncertainty (σ)': '{:.2f}',
            'Conservative (μ - 2σ)': '{:.2f}',
            'Top %': '{:.1f}%',
            'Win Rate (%)': '{:.1f}%'
        })
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Visualization comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of conservative ratings
            fig_bar = self.viz.plot_dog_comparison_bars(comparison_df)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Radar chart of multiple metrics
            fig_radar = self.viz.plot_dog_comparison_radar(comparison_df)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Head-to-head race history
        st.subheader("🏁 Head-to-Head Race History")
        h2h_races = self.get_head_to_head_races(df, comparison_df['dog_name'].tolist())
        
        if not h2h_races.empty:
            st.write(f"Found {len(h2h_races)} races where these dogs competed against each other:")
            
            # Group by race and show results
            for race_id, race_data in h2h_races.groupby('market_id'):
                race_info = race_data.iloc[0]
                st.write(f"**{race_info['race_name']} at {race_info['venue']}** - {race_info['race_time'].strftime('%Y-%m-%d')}")
                
                race_results = race_data[['dog_name', 'trap_number', 'is_winner']].copy()
                race_results['Result'] = race_results.apply(
                    lambda x: '🥇 Winner' if x['is_winner'] else f'T{x["trap_number"]}', 
                    axis=1
                )
                
                cols = st.columns(len(race_results))
                for i, (_, dog) in enumerate(race_results.iterrows()):
                    with cols[i]:
                        st.metric(dog['dog_name'], dog['Result'])
                
                st.divider()
        else:
            st.info("These dogs haven't raced against each other in the available data.")
        
        # Performance summary
        st.subheader("📈 Performance Summary")
        
        perf_data = []
        for _, dog in comparison_df.iterrows():
            dog_races = df[df['dog_name'] == dog['dog_name']]
            if not dog_races.empty:
                wins = dog_races['is_winner'].sum()
                total_races = len(dog_races)
                win_rate = wins / total_races
                venues = len(dog_races['venue'].unique())
                
                perf_data.append({
                    'Dog': dog['dog_name'],
                    'Wins': wins,
                    'Total Races': total_races,
                    'Win Rate': f"{win_rate:.1%}",
                    'Venues': venues
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    def get_head_to_head_races(self, df: pd.DataFrame, dog_names: list) -> pd.DataFrame:
        """Find races where multiple dogs from the comparison list competed."""
        # Get all races for the dogs
        dog_races = df[df['dog_name'].isin(dog_names)]
        
        # Find races with multiple dogs from our list
        race_counts = dog_races.groupby('market_id')['dog_name'].nunique()
        multi_dog_races = race_counts[race_counts >= 2].index
        
        return dog_races[dog_races['market_id'].isin(multi_dog_races)].sort_values('race_time', ascending=False)
    
    def render_betting_tool(self, df: pd.DataFrame, leaderboard: pd.DataFrame):
        """Render the Kelly Criterion betting tool."""
        st.header("💰 Kelly Criterion Betting Tool")
        
        st.info("""
        **How it works:** This tool uses TrueSkill ratings to estimate win probabilities, 
        then applies the Kelly Criterion to calculate optimal bet sizes.
        """)
        
        # Input section for race setup
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏁 Set Up Race")
            race_dogs_input = st.text_area(
                "Enter dogs in the race (one per line):",
                placeholder="Acomb Felix\nSwift Hostile\nProper Heiress\nRapid Fire\nSpeed Demon",
                height=150,
                help="Enter the names of all dogs in the race you want to analyze"
            )
        
        with col2:
            st.subheader("💷 Market Odds")
            odds_input = st.text_area(
                "Enter decimal odds (one per line):",
                placeholder="2.5\n3.2\n4.0\n8.5\n12.0",
                height=150,
                help="Enter the decimal odds for each dog (in same order as dogs)"
            )
        
        if race_dogs_input and odds_input:
            # Parse inputs
            race_dogs = [dog.strip() for dog in race_dogs_input.split('\n') if dog.strip()]
            try:
                odds_list = [float(odds.strip()) for odds in odds_input.split('\n') if odds.strip()]
            except ValueError:
                st.error("Please enter valid decimal odds (e.g., 2.5, 3.0, 4.5)")
                return
            
            if len(race_dogs) != len(odds_list):
                st.error(f"Number of dogs ({len(race_dogs)}) must match number of odds ({len(odds_list)})")
                return
            
            if len(race_dogs) < 2:
                st.error("Please enter at least 2 dogs for analysis")
                return
            
            # Check if dogs exist in our data
            missing_dogs = [dog for dog in race_dogs if dog not in leaderboard['dog_name'].values]
            if missing_dogs:
                st.warning(f"⚠️ Dogs not found in ratings: {', '.join(missing_dogs)}")
                st.info("Dogs not in our database will be assigned default ratings (new dog)")
            
            # Create odds dictionary
            market_odds = dict(zip(race_dogs, odds_list))
            
            # Get TrueSkill system from data loader
            try:
                # We need to get the TrueSkill processor from data_loader
                ts_processor = self.data_loader.get_trueskill_processor()
                if ts_processor is None:
                    st.error("TrueSkill ratings not available. Please load race data first.")
                    return
                
                # Initialize Kelly calculator
                kelly_calc = KellyCriterion(ts_processor.trueskill_system)
                
                # Get betting parameters
                bankroll = st.session_state.get('bankroll', 1000)
                max_fraction = st.session_state.get('max_bet_fraction', 0.25)
                
                # Analyze betting opportunities
                with st.spinner("Calculating optimal bet sizes..."):
                    betting_analysis = kelly_calc.analyze_race_betting_opportunities(
                        race_dogs=race_dogs,
                        market_odds=market_odds,
                        bankroll=bankroll,
                        max_fraction=max_fraction
                    )
                
                # Display results
                st.subheader("📊 Betting Analysis Results")
                
                # Summary metrics
                positive_ev_bets = betting_analysis[betting_analysis['positive_ev'] == True]
                total_recommended_bet = positive_ev_bets['recommended_bet'].sum()
                total_expected_value = positive_ev_bets['expected_value'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Positive EV Bets", len(positive_ev_bets))
                with col2:
                    st.metric("Total Bet Amount", f"£{total_recommended_bet:.2f}")
                with col3:
                    st.metric("Total Expected Value", f"£{total_expected_value:.2f}")
                with col4:
                    st.metric("% of Bankroll", f"{(total_recommended_bet/bankroll)*100:.1f}%")
                
                # Detailed analysis table
                formatted_analysis = format_betting_analysis(betting_analysis)
                
                # Color code positive EV rows
                def highlight_positive_ev(row):
                    if row['Positive EV']:
                        return ['background-color: rgba(76, 175, 80, 0.2)'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = formatted_analysis.style.apply(highlight_positive_ev, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Recommendations
                if len(positive_ev_bets) > 0:
                    st.success("✅ Found positive expected value bets!")
                    
                    st.subheader("💡 Betting Recommendations")
                    for _, bet in positive_ev_bets.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{bet['dog_name']}** - Bet £{bet['recommended_bet']:.2f}")
                            with col2:
                                st.write(f"EV: £{bet['expected_value']:.2f}")
                            with col3:
                                st.write(f"Value: {bet['value']*100:.1f}%")
                else:
                    st.warning("❌ No positive expected value bets found in this race")
                    st.info("This suggests the market odds are efficient or our model disagrees with the market assessment")
                
                # Risk simulation
                if len(positive_ev_bets) > 0:
                    st.subheader("🎲 Risk Simulation")
                    
                    with st.expander("Run Monte Carlo Simulation"):
                        num_simulations = st.slider("Number of simulations:", 100, 10000, 1000, step=100)
                        
                        if st.button("Run Simulation"):
                            with st.spinner("Running simulation..."):
                                simulation_results = kelly_calc.bankroll_simulation(
                                    betting_analysis, 
                                    initial_bankroll=bankroll,
                                    num_simulations=num_simulations
                                )
                            
                            if "message" not in simulation_results:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Mean Final Bankroll", f"£{simulation_results['mean_final_bankroll']:.2f}")
                                    st.metric("Probability of Profit", f"{simulation_results['prob_profit']*100:.1f}%")
                                    st.metric("5th Percentile", f"£{simulation_results['percentile_5']:.2f}")
                                
                                with col2:
                                    st.metric("Median Final Bankroll", f"£{simulation_results['median_final_bankroll']:.2f}")
                                    st.metric("Risk of 50% Loss", f"{simulation_results['prob_loss_50pct']*100:.1f}%")
                                    st.metric("95th Percentile", f"£{simulation_results['percentile_95']:.2f}")
                            else:
                                st.warning(simulation_results["message"])
                
                # Educational content
                with st.expander("📚 Understanding the Kelly Criterion"):
                    st.markdown("""
                    **Kelly Criterion Formula:** f = (bp - q) / b
                    
                    Where:
                    - f = fraction of bankroll to bet
                    - b = odds - 1 (net odds)
                    - p = probability of winning
                    - q = probability of losing (1 - p)
                    
                    **Key Points:**
                    - Only bet when you have an edge (positive expected value)
                    - Bet size is proportional to your edge
                    - Maximizes long-term growth rate
                    - Can be aggressive - consider fractional Kelly (25-50%) for reduced risk
                    
                    **TrueSkill Integration:**
                    - Uses dog ratings (μ) as strength estimates
                    - Applies softmax to convert to win probabilities
                    - Temperature scaling accounts for rating uncertainty
                    """)
                
            except Exception as e:
                st.error(f"Error calculating betting analysis: {str(e)}")
                st.info("Make sure you have loaded race data and TrueSkill ratings are available")
    
    def render_leaderboard_section(self, df: pd.DataFrame, leaderboard: pd.DataFrame):
        """Render the leaderboard section."""
        st.header("🏆 TrueSkill Leaderboard")
        
        if leaderboard.empty:
            st.warning("No leaderboard data available.")
            return
        
        # Filter by minimum races
        min_races = st.session_state.get('min_races', 2)
        filtered_leaderboard = leaderboard[leaderboard['races'] >= min_races].copy()
        
        # Filter by venue if selected
        venue_filter = st.session_state.get('venue_filter', 'All Venues')
        if venue_filter != 'All Venues' and not df.empty:
            # Get dogs that have raced at the selected venue
            venue_dogs = df[df['venue'] == venue_filter]['dog_name'].unique()
            filtered_leaderboard = filtered_leaderboard[filtered_leaderboard['dog_name'].isin(venue_dogs)]
            
            # Update the subtitle to show venue filter
            if venue_filter:
                st.info(f"🏟️ Showing dogs that have raced at **{venue_filter}**")
        
        if filtered_leaderboard.empty:
            if venue_filter != 'All Venues':
                st.warning(f"No dogs found with minimum {min_races} races at {venue_filter}.")
            else:
                st.warning(f"No dogs found with minimum {min_races} races.")
            return
        
        # Display metrics explanation
        with st.expander("📖 Rating Metrics Explained"):
            st.markdown("""
            - **Skill (μ)**: Estimated true skill level
            - **Uncertainty (σ)**: How confident we are in the rating
            - **Conservative**: μ - 2σ (safe rating estimate)
            - **Races**: Number of races in our dataset
            """)
        
        # Display top dogs
        top_n = st.session_state.get('top_n_dogs', 20)
        st.subheader(f"Top {top_n} Dogs (min {min_races} races)")
        
        # Calculate percentiles and format the display to match comparison table
        display_df = filtered_leaderboard.head(top_n).copy()
        total_dogs = len(leaderboard)
        
        # Add percentile column
        display_df['percentile'] = ((total_dogs - display_df['rank'] + 1) / total_dogs) * 100
        
        # Add wins and last win columns
        display_df['wins'] = 0
        display_df['last_win'] = 'Unknown'
        for idx, dog in display_df.iterrows():
            dog_races = df[df['dog_name'] == dog['dog_name']]
            if not dog_races.empty:
                # Calculate wins
                wins = dog_races['is_winner'].sum()
                display_df.loc[idx, 'wins'] = wins
                
                # Find last win date
                if 'race_time' in dog_races.columns:
                    dog_races['race_time'] = pd.to_datetime(dog_races['race_time'])
                    wins_only = dog_races[dog_races['is_winner'] == True]
                    if not wins_only.empty:
                        last_win = wins_only['race_time'].max()
                        display_df.loc[idx, 'last_win'] = last_win.strftime('%d/%m/%Y')
                    else:
                        display_df.loc[idx, 'last_win'] = 'Never'
        
        # Select and reorder columns to match comparison table
        display_cols = ['dog_name', 'rank', 'percentile', 'skill', 'uncertainty', 'conservative', 'races', 'wins', 'last_win']
        display_df = display_df[display_cols].copy()
        
        # Apply exact formatting for each column (same as comparison table)
        display_df['skill'] = display_df['skill'].astype(float).round(2)
        display_df['uncertainty'] = display_df['uncertainty'].astype(float).round(2) 
        display_df['conservative'] = display_df['conservative'].astype(float).round(2)
        display_df['percentile'] = display_df['percentile'].astype(float).round(1)
        display_df['wins'] = display_df['wins'].astype(int)
        
        display_df.columns = ['Dog Name', 'Rank', 'Top %', 'Skill (μ)', 'Uncertainty (σ)', 'Conservative (μ - 2σ)', 'Races', 'Wins', 'Last Win']
        
        # Apply styling with format preservation (same as comparison table)
        styled_df = display_df.style.format({
            'Skill (μ)': '{:.2f}',
            'Uncertainty (σ)': '{:.2f}',
            'Conservative (μ - 2σ)': '{:.2f}',
            'Top %': '{:.1f}%'
        })
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
    
    def render_visualizations(self, df: pd.DataFrame, leaderboard: pd.DataFrame, venue_stats: pd.DataFrame):
        """Render the main visualizations."""
        st.header("📊 Visualizations")
        
        # Filter leaderboard
        min_races = st.session_state.get('min_races', 2)
        top_n = st.session_state.get('top_n_dogs', 20)
        filtered_leaderboard = leaderboard[leaderboard['races'] >= min_races]
        
        if filtered_leaderboard.empty:
            st.warning("No data available for visualizations.")
            return
        
        # Top dogs chart
        st.subheader("🏃 Top Dogs by Conservative Rating")
        fig_top_dogs = self.viz.plot_top_dogs_rating(filtered_leaderboard, top_n)
        st.plotly_chart(fig_top_dogs, use_container_width=True)
        
        # Two column layout for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Skill vs Uncertainty")
            fig_scatter = self.viz.plot_skill_vs_uncertainty(filtered_leaderboard)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.subheader("📈 Rating Distribution")
            fig_dist = self.viz.plot_rating_distribution(filtered_leaderboard)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Venue analysis
        if not venue_stats.empty:
            st.subheader("🏟️ Venue Performance")
            fig_venues = self.viz.plot_venue_performance(venue_stats)
            st.plotly_chart(fig_venues, use_container_width=True)
        
        # Time series (if date data available)
        if 'race_time' in df.columns:
            st.subheader("📅 Race Activity Over Time")
            fig_time = self.viz.plot_time_series_races(df)
            st.plotly_chart(fig_time, use_container_width=True)
    
    def render_detailed_analysis(self, df: pd.DataFrame, leaderboard: pd.DataFrame):
        """Render detailed analysis section."""
        st.header("🔍 Detailed Analysis")
        
        # Tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Dog Activity", "Track Analysis", "Data Explorer"])
        
        with tab1:
            st.subheader("Dog Activity Analysis")
            min_races_activity = st.slider("Minimum races for activity analysis", 1, 10, 3)
            fig_activity = self.viz.plot_dog_activity(df, min_races_activity)
            st.plotly_chart(fig_activity, use_container_width=True)
        
        with tab2:
            st.subheader("Track Performance Comparison")
            fig_tracks = self.viz.plot_track_comparison(df, leaderboard)
            st.plotly_chart(fig_tracks, use_container_width=True)
        
        with tab3:
            st.subheader("Raw Data Explorer")
            
            # Venue filter
            if 'venue' in df.columns:
                venues = ['All'] + sorted(df['venue'].unique().tolist())
                selected_venue = st.selectbox("Filter by venue:", venues)
                
                if selected_venue != 'All':
                    filtered_df = df[df['venue'] == selected_venue]
                else:
                    filtered_df = df
            else:
                filtered_df = df
            
            # Display sample data
            st.write(f"Showing {len(filtered_df):,} rows")
            st.dataframe(filtered_df.head(1000), use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name=f"greyhound_data_{selected_venue.lower().replace(' ', '_') if selected_venue != 'All' else 'all'}.csv",
                mime="text/csv"
            )

def main():
    """Main function to run the dashboard."""
    dashboard = GreyhoundDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()