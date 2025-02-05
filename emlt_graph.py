import pandas as pd
import altair as alt

df = pd.read_csv(r"C:\Users\jyoji\Downloads\browser-eu-monthly-200910-202409.csv")


# Ensure 'Date' is parsed correctly
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')

# Melt the dataframe to long format
df_melted = df.melt(id_vars=['Date'], var_name='Browser', value_name='Share')

# Create the streamgraph using Altair
streamgraph = alt.Chart(df_melted).mark_area().encode(
    x='Date:T',
    y='Share:Q',
    color='Browser:N'
).properties(
    width=900,
    height=500,
    title="Browser Market Share Over Time"
).configure_title(
    fontSize=18
)

# Save the streamgraph as an interactive HTML file
streamgraph.save('streamgraph.html')

print("Streamgraph saved as 'streamgraph.html'")

