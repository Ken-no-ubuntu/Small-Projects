import json
import pandas as pd
import plotly as px
with open(r"C:\Users\jyoji\Downloads\jp.json", encoding='utf-8') as f:
    map_data = json.load(f)
df = pd.read_excel(r"C:\Users\jyoji\Downloads\FEH_00200502_240813125814.xlsx", usecols='B:AK', skiprows=8,sheet_name='1')
geojson_properties = [feature['properties'] for feature in map_data['features']]
geo_df = pd.DataFrame(geojson_properties)
geo_df.rename(columns={'name':'AREA'}, inplace=False)
merged_df = pd.merge(df, geo_df, on='AREA', how='right')
print(merged_df.loc[5])
import plotly.express as px

# Example: Plotting a choropleth map using a new column from the merged DataFrame
fig = px.choropleth(merged_df,
                    geojson=map_data,
                    locations='AREA',  # Column in merged_df that matches the GeoJSON properties
                    featureidkey='id',  # This should match the key in your GeoJSON
                    color='#D0330703_Health expenditure per capita (Prefecture + Municipality)[thousand yen]',  # Replace with the appropriate column from your merged DataFrame
                    color_continuous_scale="OrRd",
                    scope="asia")

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title="Average Prices by Prefecture in Japan")
fig.show()

#print(id_map)