import altair as alt
import pandas as pd

# Prepare the data
data = {
    'Date': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'USD Class I Dist': [26.91, 7.2, -12.29, 24.3, 8.75, 18.94, 18.74, 9, -8.4, 10.1],
    'Index': [18.13, 6.33, -6.83, 20.14, 1.71, 22.67, 14.87, 17.51, -6.14, 3.58]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Melt the DataFrame to long format for Altair
df_melted = df.melt('Date', var_name='Category', value_name='Value')

# Create the streamgraph
streamgraph = alt.Chart(df_melted).mark_area().encode(
    x='Date:O',
    y=alt.Y('Value:Q', stack='center'),
    color='Category:N'
).properties(
    width=600,
    height=400,
    title='Streamgraph of USD Class I Dist and Index Over Time'
)

streamgraph.show()
streamgraph.save('streamgraph_new.html')