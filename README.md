# 🐕 Greyhound TrueSkill Analytics Dashboard

An interactive Streamlit dashboard for analyzing greyhound racing performance using the [TrueSkill](https://trueskill.org/) rating system. Built to analyze UK greyhound racing data from Betfair historical files.

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

### 🏟️ **Venue Performance Analysis**
- Track-by-track performance metrics
- Average winner skill by venue
- Comprehensive venue statistics

### 🌐 Live Demo

Visit the live dashboard: **[Greyhound TrueSkill](https://greyhound-truskill.streamlit.app/)**

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

---

**Made to help Brontë gamble on her 30th**
