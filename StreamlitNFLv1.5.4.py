#!/usr/bin/env python
# coding: utf-8

# In[29]:


import streamlit as st
import json
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# Load data from JSON files
with open("data2.json", "r") as f:
    data = json.load(f)

with open("Win_Loss_all_years.json", "r") as f:
    win_loss_data = json.load(f)

# Function to extract team stats
def get_team_stats(team_name):
    stats = {}
    try:
        # Extract data for the specified team
        team_data = data[team_name]

        # Basic Team Stats
        stats["Overall W/L Record"] = win_loss_data.get(team_name, {}).get(
            "overall", "N/A"
        )  # Get overall win-loss record
        stats["Average Rushing Yards"] = round(float(
            team_data["rush_off_general"]["avg_yard"]
        ), 2)  # Convert to float and round
        stats["Average Passing Yards"] = round(float(team_data["pass_off_general"]["avg_yard"]), 2)
        stats["Average Total Yards"] = round(
            stats["Average Rushing Yards"] + stats["Average Passing Yards"], 2
        )  # Calculate total yards and round
        stats["Red Zone Efficiency"] = round(float(
            team_data["rush_off_general"]["rz_effic"]
        ) + float(
            team_data["pass_off_general"]["rz_effic"]
        ), 2)  # Sum of rushing and passing RZ efficiency and round
        stats["Turnover Percentage"] = round(( 
            float(team_data["rush_off_general"]["turnover_pct"])
            + float(team_data["pass_off_general"]["turnover_pct"])
        ) / 2, 2)  # Average of rushing and passing turnovers and round

    except KeyError:
        st.error(f"Team '{team_name}' not found in the dataset.")
        return None
    except (ValueError, TypeError) as e:
        st.error(f"Error processing data for team '{team_name}': {e}")
        return None

    return stats

# Streamlit app
def main():
    st.title("NFL Team Head-to-Head Comparison")

    # Team selection
    team1 = st.selectbox("Select Team 1", list(data.keys()))
    team2 = st.selectbox("Select Team 2", list(data.keys()))

    # Display head-to-head stats
    if team1 and team2:
        st.write(f"### Head-to-Head W/L Record ({team1} vs {team2}):")
        head_to_head_wins = (
            win_loss_data.get(team1, {})
            .get(team2, {})
            .get("wins", "Data Not Available")
        )
        head_to_head_losses = (
            win_loss_data.get(team1, {})
            .get(team2, {})
            .get("losses", "Data Not Available")
        )
        st.write(f"Wins: {head_to_head_wins}, Losses: {head_to_head_losses}")

        # Get team stats
        stats1 = get_team_stats(team1)
        stats2 = get_team_stats(team2)

        if stats1 is None or stats2 is None:
            return

        # Radar plot for average yards
        categories = ["Average Rushing Yards", "Average Passing Yards", "Average Total Yards"]
        values1 = [stats1[cat] for cat in categories]
        values2 = [stats2[cat] for cat in categories]

        # Close the radar chart to form a complete shape
        values1.append(values1[0])
        values2.append(values2[0])
        categories.append(categories[0])

        radar_fig = go.Figure()

        # Add traces for each team with adjusted opacity
        radar_fig.add_trace(
            go.Scatterpolar(
                r=values1,
                theta=categories,
                fill="toself",
                name=team1,
                opacity=0.7,  # Make the fill semi-transparent
            )
        )
        radar_fig.add_trace(
            go.Scatterpolar(
                r=values2,
                theta=categories,
                fill="toself",
                name=team2,
                opacity=0.7,  # Make the fill semi-transparent
            )
        )

        # Update radar chart layout
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(values1), max(values2)) + 10],  # Dynamic range
                )
            ),
            title=f"Comparison of Average Yards: {team1} vs {team2}",
            showlegend=True,
        )

        # Display radar chart
        st.plotly_chart(radar_fig)

        # Supplemental bar chart to highlight differences
        bar_categories = ["Average Rushing Yards", "Average Passing Yards", "Average Total Yards"]
        differences = [stats1[cat] - stats2[cat] for cat in bar_categories]

        bar_fig = px.bar(
            x=bar_categories,
            y=differences,
            color=["green" if diff > 0 else "red" for diff in differences],
            title=f"Statistical Differences ({team1} - {team2})",
            labels={"x": "Categories", "y": "Difference"},
        )
        bar_fig.update_layout(
            xaxis_title="Stat Category",
            yaxis_title=f"Difference (Positive favors {team1}, Negative favors {team2})",
            showlegend=False,
        )

        st.plotly_chart(bar_fig)

        # Display average rushing, passing, and total yards KPIs below radar chart
        st.write("### Average Yards KPIs")
        for stat_name in ["Average Rushing Yards", "Average Passing Yards", "Average Total Yards"]:
            value1 = stats1[stat_name]
            value2 = stats2[stat_name]

            # Determine colors for better and worse values
            if value1 > value2:
                stat1_color = "green"
                stat2_color = "red"
            elif value1 < value2:
                stat1_color = "red"
                stat2_color = "green"
            else:
                stat1_color = stat2_color = "black"  # Tie case

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<button style='background-color:{stat1_color}; color:white; border:none; padding:10px; width:100%; text-align:center;'>{stat_name} ({team1}): {value1}</button>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"<button style='background-color:{stat2_color}; color:white; border:none; padding:10px; width:100%; text-align:center;'>{stat_name} ({team2}): {value2}</button>",
                    unsafe_allow_html=True,
                )

        # Additional visuals for other KPIs
        st.write("### Red Zone Efficiency and Turnover Percentage")
        kpi_categories = ["Red Zone Efficiency", "Turnover Percentage"]
        kpi_values1 = [stats1[cat] for cat in kpi_categories]
        kpi_values2 = [stats2[cat] for cat in kpi_categories]

        for i, kpi_category in enumerate(kpi_categories):
            value1 = kpi_values1[i]
            value2 = kpi_values2[i]

            # Determine colors for better and worse values
            if value1 > value2:
                stat1_color = "green"
                stat2_color = "red"
            elif value1 < value2:
                stat1_color = "red"
                stat2_color = "green"
            else:
                stat1_color = stat2_color = "black"  # Tie case

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"<button style='background-color:{stat1_color}; color:white; border:none; padding:10px; width:100%; text-align:center;'>{kpi_category} ({team1}): {value1}</button>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"<button style='background-color:{stat2_color}; color:white; border:none; padding:10px; width:100%; text-align:center;'>{kpi_category} ({team2}): {value2}</button>",
                    unsafe_allow_html=True,
                )

        # Donut chart for yard composition
        st.write("### Yard Composition")

        # Create two columns for side-by-side donut charts
        col1, col2 = st.columns(2)

        for idx, (team, stats) in enumerate([(team1, stats1), (team2, stats2)]):
            yard_data = {
                "Type": ["Rushing Yards", "Passing Yards"],
                "Yards": [
                    stats["Average Rushing Yards"],
                    stats["Average Passing Yards"],
                ],
                "Raw Yards": [
                    stats["Average Rushing Yards"] * 10,  # Assume raw yards are scaled for display
                    stats["Average Passing Yards"] * 10,
                ],
            }

            donut_fig = px.pie(
                yard_data,
                values="Yards",
                names="Type",
                hole=0.4,
                title=f"{team} Yard Composition",
                hover_data=["Raw Yards"],
                labels={"Yards": "Percentage", "Raw Yards": "Raw Yards"},
            )

            if idx == 0:
                with col1:
                    st.plotly_chart(donut_fig)
            else:
                with col2:
                    st.plotly_chart(donut_fig)
     # Load the processed NFL data
    nfl_data = pd.read_csv('Processed_NFL_Data.csv')

    # Prepare data for modeling
    features = [
        'avg_rushing_yards', 'avg_passing_yards', 'avg_rush_def_yards', 'avg_pass_def_yards',
        'opponent_avg_rushing_yards', 'opponent_avg_passing_yards',
        'opponent_avg_rush_def_yards', 'opponent_avg_pass_def_yards'
    ]
    X = nfl_data[features]
    y = (nfl_data['wins'] > nfl_data['losses']).astype(int)  # 1 if more wins than losses, else 0

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a predictive model (Random Forest Classifier)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Model accuracy for verification
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Streamlit enhancements
    # Use the existing dropdowns from the original Streamlit app for team selection
    #team1 = st.session_state.get('team1')  # Assuming this is set in the original app
    #team2 = st.session_state.get('team2')  # Assuming this is set in the original app

    if team1 and team2:
        # Extract data for the selected teams
        team_1_data = nfl_data[nfl_data['team'] == team1].iloc[0]
        team_2_data = nfl_data[nfl_data['team'] == team2].iloc[0]

        # Combine data for prediction
        prediction_data = pd.DataFrame({
            'avg_rushing_yards': [team_1_data['avg_rushing_yards']],
            'avg_passing_yards': [team_1_data['avg_passing_yards']],
            'avg_rush_def_yards': [team_1_data['avg_rush_def_yards']],
            'avg_pass_def_yards': [team_1_data['avg_pass_def_yards']],
            'opponent_avg_rushing_yards': [team_2_data['avg_rushing_yards']],
            'opponent_avg_passing_yards': [team_2_data['avg_passing_yards']],
            'opponent_avg_rush_def_yards': [team_2_data['avg_rush_def_yards']],
            'opponent_avg_pass_def_yards': [team_2_data['avg_pass_def_yards']]
        })

        # Predict the winner
        prediction = model.predict(prediction_data)
        confidence = model.predict_proba(prediction_data).max()

        # Normalize confidence to reduce overconfidence
        normalized_confidence = max(0.5, min(confidence, 0.95))
        
        winner = team1 if prediction[0] == 1 else team2

        # Display prediction and confidence
        st.write(f"### Predicted Winner: {winner}")
        st.write(f"### Confidence: {confidence:.2%}")

        # Visualization: Comparative analysis
        comparison_data = pd.DataFrame({
            'Metric': ['Rushing Yards', 'Passing Yards', 'Rush Defense', 'Pass Defense'],
            team1: [
                team_1_data['avg_rushing_yards'],
                team_1_data['avg_passing_yards'],
                team_1_data['avg_rush_def_yards'],
                team_1_data['avg_pass_def_yards']
            ],
            team2: [
                team_2_data['avg_rushing_yards'],
                team_2_data['avg_passing_yards'],
                team_2_data['avg_rush_def_yards'],
                team_2_data['avg_pass_def_yards']
            ]
        })

        fig = px.bar(comparison_data, x='Metric', y=[team1, team2],
                     barmode='group', title='Team Metrics Comparison')
        st.plotly_chart(fig)           
                    
if __name__ == "__main__":
    main()


# In[ ]:




