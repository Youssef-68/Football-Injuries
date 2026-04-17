Sports Injury Analytics System
Project Overview

A comprehensive data analytics and machine learning web application for analyzing sports injuries, identifying patterns, and predicting recovery times. Built with Streamlit, this system helps medical staff, coaches, and sports analysts make data-driven decisions about injury management.
Features
1. Data Cleaning & Processing

    Automatic date formatting and missing value handling

    Standardized injury categorization (40+ injury types)

    Season normalization (1989/90 to 2024/25 format)

    Outlier removal for statistical accuracy

2. Interactive Filters

    Season Filter: Select specific seasons (89/90 to 24/25)

    Injury Type Filter: Filter by 40+ categorized injury types

    Date Range Filter: Fixed range 01/07/1989 to 30/06/2025

    Real-time filter validation with user feedback

3. Analytics Dashboard (5 Tabs)
Tab 1: Injuries Analysis

    Top 10 injury types visualization

    Injuries per season trend

    Multi-line trends for top 8 injury types

    AI-generated insights for each chart

Tab 2: Timeline Analysis

    Monthly injury distribution (July to June season)

    Average injury duration over years

    Start vs end months comparison

    Short vs long injury distribution

    Seasonal pattern detection

Tab 3: Days Missed Analysis

    Distribution histogram with outliers removed

    Boxplot visualization

    Average days missed by injury type

    Season severity trends

    Statistical insights

Tab 4: Games Missed Analysis

    Distribution analysis

    Boxplot with outlier removal

    Average games missed by injury type

    Seasonal trend analysis

Tab 5: Injury Predictor (ML Model)

    Predict injury duration based on:

        Injury type

        Month of injury

        Day of week

        Player history (avg games missed, avg duration, injury count)

        Last injury severity

    Real-time predictions with confidence scores

    Similar historical cases comparison

    Player history impact analysis

4. Machine Learning Models
Regression Model (XGBoost)

    Purpose: Predict exact days missed

    Performance: MAE: ~12 days, RMSE: ~60 days

    Features: 15+ injury and player-related features

    Target: Log-transformed days missed

Classification Model (XGBoost)

    Purpose: Classify injury severity

    Classes:

        Class 0: Short (≤7 days)

        Class 1: Medium (8-30 days)

        Class 2: Long (31-90 days)

        Class 3: Severe (>90 days)

    Performance: 86% accuracy with balanced class weights

5. AI Insights Engine

Rule-based intelligence providing automatic insights:

    Trend detection (increasing/decreasing/stable)

    Distribution analysis (skewness, spread)

    Pattern recognition (seasonal, monthly)

    Comparative analysis (top/bottom performers)

    Outlier impact assessment

Technology Stack
Core Technologies

    Python 3.12: Primary programming language

    Streamlit: Web application framework

    Pandas: Data manipulation and analysis

    NumPy: Numerical computing

    Plotly: Interactive visualizations

Machine Learning

    XGBoost: Gradient boosting for regression and classification

    Scikit-learn: Train-test split, metrics, preprocessing

        LabelEncoder for categorical variables

        train_test_split for data partitioning

        classification_report, confusion_matrix for evaluation

        compute_class_weight for handling imbalance

Data Processing

    Datetime handling: Date parsing and transformation

    IQR outlier removal: Statistical outlier detection

    Feature engineering: Player-level aggregations
