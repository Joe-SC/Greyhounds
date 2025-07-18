# 🐕 Greyhound TrueSkill Analytics Dashboard

An interactive Streamlit dashboard for analyzing greyhound racing performance using the TrueSkill rating system. Built to analyze UK greyhound racing data from Betfair historical files.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## 🎯 Features

### 📊 **Interactive Dashboard**
- Real-time TrueSkill rating calculations
- Interactive charts and visualizations
- Comprehensive leaderboards with filtering

### 🔍 **Dog Search & Analysis**
- Quick lookup for specific dogs
- Detailed performance profiles
- Recent race history and form

### ⚖️ **Multi-Dog Comparison**
- Compare multiple dogs side-by-side
- Perfect for race field analysis
- Head-to-head race history
- Visual rating comparisons with radar charts

### 🏟️ **Venue Performance Analysis**
- Track-by-track performance metrics
- Average winner skill by venue
- Comprehensive venue statistics

### ⚡ **Smart Caching**
- File-based caching for faster subsequent runs
- Automatic data processing and storage
- Optimized for large datasets

## 🚀 Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/greyhounds.git
   cd greyhounds
   ```

2. **Install dependencies with Poetry:**
   ```bash
   poetry install
   poetry shell
   ```

3. **Run the dashboard:**
   ```bash
   python run_dashboard.py
   ```

4. **Open your browser to:** `http://localhost:8501`

### 🌐 Live Demo

Visit the live dashboard: [**Greyhound TrueSkill Analytics**](https://your-app-name.streamlit.app)

## 📁 Project Structure

```
greyhounds/
├── analytics/                  # Dashboard modules
│   ├── __init__.py
│   ├── dashboard.py           # Main Streamlit app
│   ├── data_loader.py         # Data loading & caching
│   └── visualizations.py     # Plotly charts
├── betfair_parser.py          # Betfair data parsing
├── greyhound_trueskill.py     # TrueSkill implementation
├── run_dashboard.py           # Dashboard runner script
├── demo.ipynb               # Analysis demonstration
├── race_data_*.csv          # Race data files
└── cache/                   # Cached calculations
```

## 🎮 How to Use

### 🐕 **Dog Lookup**
1. Enter a dog's name in the sidebar search box
2. View detailed ratings, recent form, and race history
3. Perfect for researching dogs in upcoming races

### ⚖️ **Compare Dogs** (Race Analysis)
1. Enter multiple dog names in the "Compare Dogs" text area (one per line)
2. Example:
   ```
   Acomb Felix
   Swift Hostile
   Proper Heiress
   ```
3. Get side-by-side comparison with:
   - TrueSkill ratings and rankings
   - Win rates and experience
   - Head-to-head race history
   - Visual comparisons

### 📊 **Explore Data**
- Use filters to adjust minimum race requirements
- Browse venue performance analysis
- Download filtered data as CSV

## 📈 Understanding TrueSkill Ratings

- **Skill (μ)**: Estimated true skill level
- **Uncertainty (σ)**: Confidence in the rating
- **Conservative**: μ - 2σ (safe rating estimate)
- **Races**: Number of races in dataset

Higher skill and conservative ratings indicate better-performing dogs. Lower uncertainty means more confidence in the rating.

## 🏁 Perfect for Race Analysis

This dashboard is ideal for:
- **Pre-race analysis**: Compare all dogs in an upcoming race
- **Form study**: Research individual dog performance
- **Track analysis**: Understand venue-specific performance
- **Historical research**: Explore head-to-head matchups

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Interactive web dashboard
- **TrueSkill**: Rating system algorithm
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **BetfairLightweight**: Betfair API integration

### Data Processing
- Processes Betfair historical race data
- Calculates TrueSkill ratings using Bayesian inference
- Handles race results, trap positions, and timestamps
- Caches calculations for performance

### Performance
- Smart caching reduces subsequent load times
- Handles datasets with 25,000+ races
- Optimized for interactive exploration

## 📝 Data Requirements

The dashboard works with:
- **Betfair historical data files** (BASIC format)
- **Processed CSV files** from the analysis pipeline
- **Date range**: 2025-01-01 to present (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data usage terms.

## 🎉 Acknowledgments

- Built with the [TrueSkill algorithm](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) developed by Microsoft Research
- Designed for UK greyhound racing analysis
- Powered by Streamlit for interactive data exploration

---

**Made with ❤️ for greyhound racing analytics**