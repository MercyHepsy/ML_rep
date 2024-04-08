import streamlit as st
from snowflake.snowpark import Session
import pandas as pd
import plotly.express as px

session = Session.builder.getOrCreate()

data = session.table("HOSPITAL.READMISSION.BEDS_DATA1")
list = data.collect()
df =  pd.DataFrame(list)

# Page Layout
c1,c2,c3=st.columns([0.2,0.60,0.2])
with c2:
    st.title("Hospital Re-Admission Analysis")

chart_df = df.copy()

# Page Layout
c1, c2, c3, c4, c5  = st.columns([2.5, 45, 5, 45, 2.5])

with c2:    
    # Heading
    st.write("### Rate of Patients Readmitted after 30 Days by Age Group")

    # Calculate the percentage of patients readmitted within 30 days by age group
    readmitted_after_30_days = chart_df[chart_df['READMITTED'] == '>30days']
    readmitted_counts_by_age = readmitted_after_30_days.groupby('AGE').size()
    total_counts_by_age = chart_df.groupby('AGE').size()
    readmission_rates_by_age = ((readmitted_counts_by_age / total_counts_by_age) * 100).round(2)

    # Create a Barchart plotting for percentage of patients readmitted within 30 days by age group
    readmission_rates_age = pd.DataFrame({'Age Group': readmission_rates_by_age.index,
                                        'Readmission Rate (%)': readmission_rates_by_age.values})

    # Sort the DataFrame by age group
    readmission_rates_age = readmission_rates_age.sort_values(by='Age Group')

    # Plot the bar chart
    fig = px.bar(readmission_rates_age, x='Age Group', y='Readmission Rate (%)',
                labels={'Readmission Rate (%)': 'Readmission Rate (%)'},
                text='Readmission Rate (%)',
                color='Age Group', width=600)
    
    # Set x-axis range
    fig.update_xaxes(range=[10, 100]) 
    
    # Set transparent background
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
    
    # Customize tick and label colors for x-axis and y-axis
    fig.update_xaxes(tickfont=dict(color='#5CA8F1'),  # Change x-axis tick color to blue
                    titlefont=dict(color='#EF7B45'))  # Change x-axis label color to blue
    fig.update_yaxes(tickfont=dict(color='#5CA8F1'),  # Change y-axis tick color to green
                    titlefont=dict(color='#EF7B45'))  # Change y-axis label color to blue

    # show chart
    st.plotly_chart(fig, use_container_width=True)
