import streamlit as st

from streamlit_jupyter import StreamlitPatcher, tqdm
import os
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import squarify
import seaborn as sns
import folium
from pyproj import CRS
from streamlit_folium import folium_static
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.header("Our Research")

tab1, tab2, tab3 = st.tabs(
    ["Trend of crimes & Impact of nuisances", "Crime Rate vs Region in Breda", "Crime Rate vs Season"])

with tab1:
    st.title('Trend of crimes & Impact of nuisances                           ')

    livability_index = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/livability_index (1).csv')
    oops = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/breda_grid_keys.gpkg')

    boa_reg_2018 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2018.shp')

    boa_reg_2019 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2019.shp')

    boa_reg_2020 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2020.shp')

    boa_reg_2021 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2021.shp')

    boa_reg_2022 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2022.shp')
    breda_full_grid = gpd.read_file(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/Breda-Full_CBS-Grid.shp')


    def translate_name(crime):

        if crime == 'Fietsoverlast':

            return 'Bicycle Nuisance'

        elif crime == 'Parkeeroverlast':

            return 'Parking Nuisance'

        elif crime == 'Wildplassen':

            return 'Pee in public'

        elif crime == 'Afval':

            return 'Waste'

        elif crime == 'Jeugd':

            return 'Youth Nuisances'

        elif crime == 'Drugs/alcohol':

            return 'Drugs/Alcohol Nuisances'

        elif crime == 'Zwervers':

            return 'Homeless People'

        else:

            return ''


    def rename_column(df, column_name, new_name):

        df = df.rename(columns={column_name: new_name})

        return df


    boa_reg_2018.Soort = boa_reg_2018.Soort.apply(lambda x: translate_name(x))
    boa_reg_2018 = rename_column(boa_reg_2018, 'CBS_code', 'cbs_grid_code')

    boa_reg_2019.Soort = boa_reg_2019.Soort.apply(lambda x: translate_name(x))
    boa_reg_2019 = rename_column(boa_reg_2019, 'CBS_code', 'cbs_grid_code')

    boa_reg_2020.Soort = boa_reg_2020.Soort.apply(lambda x: translate_name(x))
    boa_reg_2020 = rename_column(boa_reg_2020, 'CBS_code', 'cbs_grid_code')

    boa_reg_2021.Soort = boa_reg_2021.Soort.apply(lambda x: translate_name(x))
    boa_reg_2021 = rename_column(boa_reg_2021, 'CBS_code', 'cbs_grid_code')

    boa_reg_2022.Soort = boa_reg_2022.Soort.apply(lambda x: translate_name(x))
    boa_reg_2022 = rename_column(boa_reg_2022, 'CBS_code', 'cbs_grid_code')

    nusiances_2018 = pd.merge(oops, boa_reg_2018, on='cbs_grid_code', how='outer')
    nusiances_2019 = pd.merge(oops, boa_reg_2019, on='cbs_grid_code', how='outer')
    nusiances_2020 = pd.merge(oops, boa_reg_2020, on='cbs_grid_code', how='outer')
    nusiances_2021 = pd.merge(oops, boa_reg_2021, on='cbs_grid_code', how='outer')
    nusiances_2022 = pd.merge(oops, boa_reg_2022, on='cbs_grid_code', how='outer')

    livability_index_grid = pd.merge(oops, livability_index, on='cbs_grid_code', how='outer')

    livability_index_grid.crs = CRS.from_epsg(4326)
    livability_index_grid = livability_index_grid.to_crs(epsg=4326)
    livability_index_grid['geoid'] = livability_index_grid.index.astype('str')
    livability_index_grid = livability_index_grid[['geoid', 'livability_score', 'geometry']]
    livability_index_grid = livability_index_grid.dropna(subset=['livability_score'])

    dtype_mapping = {
        'Column1': float,
        'Column2': int,
        # Specify data types for other columns as needed
    }

    recorded_crimes_Breda_per_month = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/Processed agony.csv',
                                                  dtype=dtype_mapping, low_memory=False)
    recorded_crimes_Breda_per_month = recorded_crimes_Breda_per_month.rename(
        columns={'Soort misdrijf': 'Type of Crime'})
    recorded_crimes_Breda_per_month = recorded_crimes_Breda_per_month.rename(columns={'Perioden': 'Periods'})
    recorded_crimes_Breda_per_month = recorded_crimes_Breda_per_month.rename(
        columns={'Wijken en buurten': 'Districts and neighbourhoods'})
    recorded_crimes_Breda_per_month = recorded_crimes_Breda_per_month.rename(
        columns={'Geregistreerde misdrijven (aantal)': 'Registered Crimes (total)'})
    ColumnsTokeep = ['Type of Crime', 'Periods', 'Districts and neighbourhoods', 'Registered Crimes (total)']
    TotalCrimes = recorded_crimes_Breda_per_month.drop(
        columns=[col for col in recorded_crimes_Breda_per_month.columns if col not in ColumnsTokeep])
    TotalcrimesBreda = TotalCrimes.drop(TotalCrimes.index[136:])
    TotalcrimesBreda['Registered Crimes (total)'] = TotalcrimesBreda['Registered Crimes (total)'].astype(float)


    @st.cache_data
    def create_graph(df):
        # Creating figure to make the plot
        fig = go.Figure(data=go.Scatter(x=TotalcrimesBreda['Periods'], y=TotalcrimesBreda['Registered Crimes (total)']))
        fig.update_layout(
            xaxis_title='Months',
            yaxis_title='Number of Registered Crimes'
        )
        return fig


    # Readying the streamlit
    def main():
        # show_graph = st.button("Show Graph on Total Registered Crimes in Breda")
        # if show_graph:
        st.subheader("Total Registered Crimes in Breda per month 2012 - 2023")

        # Plotting the graph
        fig = create_graph(TotalcrimesBreda)

        # Displaying the graph
        st.plotly_chart(fig, use_container_width=True,
                        config={'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d']})
        # Explaining the graph
        st.write(
            "This graph shows the amount of crimes that were registered in Breda from 2012 Januari till 2023 of April. as you can see from the visualization there are more negative correlation across the years with registered crimes but there are some months that has a drastic increase in registred crimes.(per example 2013 Januari there was a massive increase in registred crimes). my findings are:")
        st.write("- On Jan 2013 was the highest amount of registered crimes in Breda")
        st.write("- On Jan 2021 was the lowest amount of registered crimes in Breda")


    # else:
    # st.info("click this button to see the Total Crimes from 2012 Jan to 2023 Apr")

    if __name__ == "__main__":
        main()


    def sort_by_total_number(df):

        df['Total'] = df.sum(numeric_only=True, axis=1)

        df = df.sort_values(by=['Total'], ascending=False)

        return df


    def get_total_number_of_crimes_per_type(df):

        return df.Total.sum()


    boa_reg_2018.Soort = boa_reg_2018.Soort.apply(lambda x: translate_name(x))
    boa_reg_2018 = rename_column(boa_reg_2018, 'CBS_code', 'cbs_grid_code')

    boa_reg_2019.Soort = boa_reg_2019.Soort.apply(lambda x: translate_name(x))
    boa_reg_2019 = rename_column(boa_reg_2019, 'CBS_code', 'cbs_grid_code')

    boa_reg_2020.Soort = boa_reg_2020.Soort.apply(lambda x: translate_name(x))
    boa_reg_2020 = rename_column(boa_reg_2020, 'CBS_code', 'cbs_grid_code')

    boa_reg_2021.Soort = boa_reg_2021.Soort.apply(lambda x: translate_name(x))
    boa_reg_2021 = rename_column(boa_reg_2021, 'CBS_code', 'cbs_grid_code')

    boa_reg_2022.Soort = boa_reg_2022.Soort.apply(lambda x: translate_name(x))
    boa_reg_2022 = rename_column(boa_reg_2022, 'CBS_code', 'cbs_grid_code')

    oops = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/cbs_breda_grid.gpkg')

    nusiances_2018 = pd.merge(oops, boa_reg_2018, on='cbs_grid_code', how='outer')
    nusiances_2019 = pd.merge(oops, boa_reg_2019, on='cbs_grid_code', how='outer')
    nusiances_2020 = pd.merge(oops, boa_reg_2020, on='cbs_grid_code', how='outer')
    nusiances_2021 = pd.merge(oops, boa_reg_2021, on='cbs_grid_code', how='outer')
    nusiances_2022 = pd.merge(oops, boa_reg_2022, on='cbs_grid_code', how='outer')


    def get_data_ready_for_map(df):
        df = df.rename(columns={'geometry_x': 'geometry'})
        df.crs = CRS.from_epsg(4326)
        df = df.to_crs(epsg=4326)
        df['geoid'] = df.index.astype('str')
        df = df[['cbs_grid_code', 'geoid', 'Count', 'geometry']]
        df = df.dropna(subset=['Count'])

        return df


    def clean_after_merge(df):

        df['Count_y'] = df['Count_y'].fillna(0)

        df['Count'] = df['Count_x'] + df['Count_y']

        df['geoid'] = df['geoid_x']

        df['geometry'] = df['geometry_x']

        df = df[['cbs_grid_code', 'geoid', 'geometry', 'Count']]

        return df


    nusiances_2018 = get_data_ready_for_map(nusiances_2018)
    nusiances_2019 = get_data_ready_for_map(nusiances_2019)
    nusiances_2020 = get_data_ready_for_map(nusiances_2020)
    nusiances_2021 = get_data_ready_for_map(nusiances_2021)
    nusiances_2022 = get_data_ready_for_map(nusiances_2022)

    nusiances_total = pd.merge(nusiances_2018, nusiances_2019, on='cbs_grid_code', how='outer')
    nusiances_total = clean_after_merge(nusiances_total)

    nusiances_total = pd.merge(nusiances_total, nusiances_2020, on='cbs_grid_code', how='outer')
    nusiances_total = clean_after_merge(nusiances_total)

    nusiances_total = pd.merge(nusiances_total, nusiances_2021, on='cbs_grid_code', how='outer')
    nusiances_total = clean_after_merge(nusiances_total)

    nusiances_total = pd.merge(nusiances_total, nusiances_2022, on='cbs_grid_code', how='outer')
    nusiances_total = clean_after_merge(nusiances_total)

    nusiances_total['geoid'] = nusiances_total.index.astype('str')
    nusiances_total = nusiances_total[['geoid', 'geometry', 'Count']]
    nusiances_total = nusiances_total.drop_duplicates(subset='geometry')

    nusiances_total['geoid'] = nusiances_total.index.astype('str')
    my_map = folium.Map(location=[51.571915, 4.768323], zoom_start=13)
    Choropleth = folium.Choropleth(geo_data=nusiances_total,
                                   name='Nuisances Total',
                                   data=nusiances_total,
                                   columns=['geoid', 'Count'],
                                   key_on='feature.id',
                                   fill_color='YlGn',
                                   fill_opacity=0.8,
                                   line_opacity=0.2,
                                   line_color='white',
                                   line_weight=0,
                                   highlight=False,
                                   smooth_factor=1.0,
                                   bins=3,
                                   reset=True
                                   ).add_to(my_map)

    Choropleth = folium.Choropleth(geo_data=livability_index_grid,
                                   name='Quality of Life',
                                   data=livability_index_grid,
                                   columns=['geoid', 'livability_score'],
                                   key_on='feature.id',
                                   fill_color='BuPu',
                                   fill_opacity=0.5,
                                   line_opacity=0.2,
                                   line_color='white',
                                   line_weight=0,
                                   highlight=False,
                                   smooth_factor=1.0,
                                   bins=3,
                                   reset=True
                                   ).add_to(my_map)

    folium.map.LayerControl().add_to(my_map)


    def main():
        # show_map = st.button("Show Map on Nuissance vs Quality of life")
        # if show_map:
        st.subheader("Nuissance vs Quality of life")

        # Display the map on Streamlit
        folium_static(my_map)
        st.write(
            "Neighbourhood Analysis Upon reviewing registrations from Breda, it becomes apparent that there is a significant variation in crime and nuisance rates among different neighborhoods. After identifying the ten neighborhoods with the highest crime rates in Breda, we further examined the occurrence of nuisances. We observed that regions experiencing high rates of crime also tend to have high levels of nuisances. This implies that an increase in the number of nuisances can sometimes lead to an uptick in crime. Additionally, we explored the correlation between the Quality of Life index and crime rate in each neighborhood. It was observed that neighborhoods with higher quality of life tend to be less susceptible to crime. Consequently, it is of utmost importance for law enforcement to assess the conditions in different neighborhoods, drawing on historical data, in order to allocate resources in a suitable and efficient manner.")


    # else:
    # st.info("click this button to see the map that shows the impact of Nuissances vs Quality of life")

    if __name__ == '__main__':
        main()

    CrimeVsQol = pd.read_csv("Municipality-AI-Powered-Web-App/finalApp/data/data/data_for_model.csv", delimiter=",",
                             skiprows=0)  # , index_col= 0
    CrimeVsQol['date'] = pd.to_datetime(CrimeVsQol['date'])
    CrimeVsQol['number_of_crimes'] = pd.to_numeric(CrimeVsQol['number_of_crimes'], errors='coerce')
    CrimeVsQol['livability_score'] = pd.to_numeric(CrimeVsQol['livability_score'], errors='coerce')


    def clean_and_convert(value):
        cleaned_value = ''.join(filter(str.isdigit, str(value)))
        if cleaned_value:
            return int(cleaned_value)
        else:
            return None


    def main():
        # show_map = st.button("Show Neighborhood Analysis")
        # if show_map:
        # Create the Streamlit app
        st.title('Neighborhood Analysis')
        st.write('Select filters to view livability score and crime numbers.')

        # Filter by neighborhood
        selected_neighborhoods = st.multiselect('Select Neighborhood(s)', CrimeVsQol['neighbourhood_name'].unique())

        # Filtering on range
        year_range = st.slider('Select Year Range', int(CrimeVsQol['date'].dt.year.min()),
                               int(CrimeVsQol['date'].dt.year.max()),
                               (int(CrimeVsQol['date'].dt.year.min()), int(CrimeVsQol['date'].dt.year.max())))
        year_range = [pd.to_datetime(str(year)) for year in year_range]

        # Creating filters
        filtered_data = CrimeVsQol[
            CrimeVsQol['neighbourhood_name'].isin(selected_neighborhoods) & CrimeVsQol['date'].between(year_range[0],
                                                                                                       year_range[
                                                                                                           1]) & (
                        CrimeVsQol['number_of_crimes'] > 0)]
        filtered_data = filtered_data.groupby('neighbourhood_name').head(9)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Livability Score", "Crime Numbers"))

        # Making the livability subplot
        if 'livability_score' in filtered_data.columns:
            for neighborhood in selected_neighborhoods:
                neighborhood_data = filtered_data[filtered_data['neighbourhood_name'] == neighborhood]
                fig.add_trace(
                    go.Scatter(x=neighborhood_data['date'], y=neighborhood_data['livability_score'], name=neighborhood),
                    row=1, col=1)

        # Adding Number of crimes to the subplot
        if 'number_of_crimes' in filtered_data.columns:
            for neighborhood in selected_neighborhoods:
                neighborhood_data = filtered_data[filtered_data['neighbourhood_name'] == neighborhood]
                fig.add_trace(
                    go.Scatter(x=neighborhood_data['date'], y=neighborhood_data['number_of_crimes'], name=neighborhood),
                    row=1, col=2)

        # Updating the layout and plotting the updated layout
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig)
        st.write(
            "With this data you can see how the frequency and number of crimes have a impact on the nieghborhoods livability score. As you start comparing the neighborhoods together per exapmple valkenberg and the station you'd see that valkenberg has a higher livability score compared to the station and you can also see that the station has a higher number of crimes compared to valkenberg. With this data you'd see the correlation between livabilty score and number of crimes over a period")


    # else:
    # st.info("Click the button to see the Neighborhood Analysis and the the impact crimes has on the neighborhoods livability score")

    if __name__ == '__main__':
        main()

with tab2:
    def rename_first_col(df):

        df = df.rename({
            'Unnamed: 0': 'Region'
        }, axis='columns')

        return df


    abuse = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/abuse_breda.csv',
                        sep=';')
    abuse = rename_first_col(abuse)

    accidents = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/accidents_breda.csv',
                            sep=';', skiprows=[0, 1, 2, 3, 5], decimal='.')
    accidents = rename_first_col(accidents)

    arms_trade = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/arms_trade_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    arms_trade = rename_first_col(arms_trade)

    discrimination = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/discrimination_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    discrimination = rename_first_col(discrimination)

    drug_trafficking = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/drug_trafficking_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    drug_trafficking = rename_first_col(drug_trafficking)

    drugs_drinks_nuisances = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/drugs_drinks_nuisances_breda.csv',
        sep=';', skiprows=[0, 1, 2, 3, 5], decimal='.')
    drugs_drinks_nuisances = rename_first_col(drugs_drinks_nuisances)

    fire_explosion = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/fire_explosion_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    fire_explosion = rename_first_col(fire_explosion)

    fireworks = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/firewroks_breda.csv',
                            sep=';', skiprows=[0, 1, 2, 3, 5], decimal='.')
    fireworks = rename_first_col(fireworks)

    human_trafficking = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/human_trafficking.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    human_trafficking = rename_first_col(human_trafficking)

    imigration_care = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/immigration_care_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    imigration_care = rename_first_col(imigration_care)

    murders = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/murders_breda.csv',
                          sep=';', skiprows=[0, 1, 2, 3, 5], decimal='.')
    murders = rename_first_col(murders)

    open_violence = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/open_violence_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    open_violence = rename_first_col(open_violence)

    people_smuggling = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/people_smuggling_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    people_smuggling = rename_first_col(people_smuggling)

    pick_pocketing = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/pick_pocketing_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    pick_pocketing = rename_first_col(pick_pocketing)

    robbery = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/robbery_breda.csv',
                          sep=';', skiprows=[0, 1, 2, 3, 5], decimal='.')
    robbery = rename_first_col(robbery)

    shoplifting = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/shoplifting_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    shoplifting = rename_first_col(shoplifting)

    street_robbery = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/street_robbery_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    street_robbery = rename_first_col(street_robbery)

    threat = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/threar_breda.csv',
                         sep=';', skiprows=[0, 1, 2, 3, 5], decimal='.')
    threat = rename_first_col(threat)

    home_theft = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/total_num_home_theft_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    home_theft = rename_first_col(home_theft)

    under_influence_road = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/under_influence_breda.csv', sep=';',
        skiprows=[0, 1, 2, 3, 5], decimal='.')
    under_influence_road = rename_first_col(under_influence_road)

    total_crimes = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/recorded_crimes_breda_reg/total_crimes_number_breda_regions.csv',
        sep=';', skiprows=[0, 1, 2, 3, 5], decimal='.')
    total_crimes = rename_first_col(total_crimes)


    def sort_by_total_number(df):

        df['Total'] = df.sum(numeric_only=True, axis=1)

        df = df.sort_values(by=['Total'], ascending=False)

        return df


    abuse = sort_by_total_number(abuse)

    accidents = sort_by_total_number(accidents)

    arms_trade = sort_by_total_number(arms_trade)

    discrimination = sort_by_total_number(discrimination)

    drug_trafficking = sort_by_total_number(drug_trafficking)

    drugs_drinks_nuisances = sort_by_total_number(drugs_drinks_nuisances)

    fire_explosion = sort_by_total_number(fire_explosion)

    fireworks = sort_by_total_number(fireworks)

    human_trafficking = sort_by_total_number(human_trafficking)

    imigration_care = sort_by_total_number(imigration_care)

    murders = sort_by_total_number(murders)

    open_violence = sort_by_total_number(open_violence)

    people_smuggling = sort_by_total_number(people_smuggling)

    pick_pocketing = sort_by_total_number(pick_pocketing)

    robbery = sort_by_total_number(robbery)

    shoplifting = sort_by_total_number(shoplifting)

    street_robbery = sort_by_total_number(street_robbery)

    threat = sort_by_total_number(threat)

    home_theft = sort_by_total_number(home_theft)

    under_influence_road = sort_by_total_number(under_influence_road)

    total_crimes = sort_by_total_number(total_crimes)


    def get_total_number_of_crimes_per_type(df):

        return df.Total.sum()


    crime_names = ['abuse', 'accidents', 'arms_trade', 'drug_trafficking', 'human_trafficking', 'murders',
                   'open_violence', 'people_smuggling', 'pick_pocketing', 'robbery', 'shoplifting', 'street_robbery']

    number_of_crimes_per_type = []

    for crime_t in [abuse, accidents, arms_trade, drug_trafficking, human_trafficking, murders, open_violence,
                    people_smuggling, pick_pocketing, robbery, shoplifting, street_robbery, threat]:
        number_of_crimes_per_type.append(get_total_number_of_crimes_per_type(crime_t))

    crime_name_num_dict = {
        crime_names[i]: number_of_crimes_per_type[i] for i in range(len(crime_names))
    }

    crime_name_num_dict = sorted(crime_name_num_dict.items(), key=lambda x: x[1])

    crime_name_num_dict = dict(crime_name_num_dict)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("", "", ""))

    fig.add_trace(go.Bar(x=list(crime_name_num_dict.keys())[0:3], y=list(crime_name_num_dict.values())[0:3]), row=1,
                  col=1)
    fig.add_trace(go.Bar(x=list(crime_name_num_dict.keys())[3:7], y=list(crime_name_num_dict.values())[3:7]), row=1,
                  col=2)
    fig.add_trace(go.Bar(x=list(crime_name_num_dict.keys())[7:], y=list(crime_name_num_dict.values())[7:]), row=1,
                  col=3)

    fig.update_layout(
        title='Comparison of selected crimes in Breda:',
        title_font=dict(size=30),
        xaxis_tickangle=-45,
        yaxis=dict(tickfont=dict(size=20)),
        legend_title='Legend',
        showlegend=False
    )

    fig.update_xaxes(tickangle=-45, tickfont=dict(size=20), row=1, col=1)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=20), row=1, col=2)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=20), row=1, col=3)

    st.plotly_chart(fig)

    crime_reg = {
        'Abuse': [reg_name for reg_name in abuse.Region.unique()[0:5]],
        'Accidents': [reg_name for reg_name in accidents.Region.unique()[0:5]],
        'Arms_Trade': [reg_name for reg_name in arms_trade.Region.unique()[0:5]],
        'Discrimination': [reg_name for reg_name in discrimination.Region.unique()[0:5]],
        'Drug_Trafficking': [reg_name for reg_name in drug_trafficking.Region.unique()[0:5]],
        'Drugs_Drinks_Nuisances': [reg_name for reg_name in drugs_drinks_nuisances.Region.unique()[0:5]],
        'Fire_Explosion': [reg_name for reg_name in fire_explosion.Region.unique()[0:5]],
        'Human_Trafficking': [reg_name for reg_name in human_trafficking.Region.unique()[0:5]],
        'Immigration_Care': [reg_name for reg_name in imigration_care.Region.unique()[0:5]],
        'Murders': [reg_name for reg_name in murders.Region.unique()[0:5]],
        'Open_Violence': [reg_name for reg_name in open_violence.Region.unique()[0:5]],
        'People_Smuggling': [reg_name for reg_name in people_smuggling.Region.unique()[0:5]],
        'Pick_Pocketing': [reg_name for reg_name in pick_pocketing.Region.unique()[0:5]],
        'Robbery': [reg_name for reg_name in robbery.Region.unique()[0:5]],
        'Shoplifting': [reg_name for reg_name in shoplifting.Region.unique()[0:5]],
        'Street_Robbery': [reg_name for reg_name in street_robbery.Region.unique()[0:5]],
        'Threat': [reg_name for reg_name in threat.Region.unique()[0:5]],
        'Home_Theft': [reg_name for reg_name in home_theft.Region.unique()[0:5]],
        'Driving_under_Influence': [reg_name for reg_name in under_influence_road.Region.unique()[0:5]],
        'Total_Number_of_Crimes': [reg_name for reg_name in total_crimes.Region.unique()[0:5]]
    }

    crimre_reg_df = pd.DataFrame(crime_reg)

    # crime_reg

    crimre_reg_df = crimre_reg_df.drop(columns=['Accidents'])


    def create_tier_list(df, number_of_regions):

        tier_list = []

        unique_regs = pd.unique(df[list(df.columns.values)].values.ravel())

        regs_score = [0] * len(unique_regs)

        city_index = 0

        for column in crimre_reg_df.columns.values:
            points = 5

            for reg in crimre_reg_df[column].values:
                regs_score[list(unique_regs).index(reg)] += points

                points -= 1

        sorted_nums = sorted(regs_score, reverse=True)

        nums_checked = []

        for num in sorted_nums:

            if list(unique_regs)[list(regs_score).index(num)] in tier_list:

                if num in nums_checked:

                    city_index += 1

                    tier_list.append(list(unique_regs)[list(np.where(np.array(regs_score) == num)[0])[city_index]])
                else:

                    nums_checked.append(num)

                    city_index = 1

                    tier_list.append(list(unique_regs)[list(np.where(np.array(regs_score) == num)[0])[city_index]])

            else:

                tier_list.append(list(unique_regs)[list(np.where(np.array(regs_score) == num)[0])[0]])

        return tier_list[0:number_of_regions], sorted_nums[0:number_of_regions]


    neighbourhood_l, number_of_crimes_l = create_tier_list(crimre_reg_df, 10)

    plt.figure(figsize=(25, 6))

    fig = go.Figure()

    fig.add_trace(go.Treemap(
        labels=neighbourhood_l,
        parents=[""] * len(neighbourhood_l),
        values=number_of_crimes_l,
        marker=dict(colors=sns.color_palette("tab20", len(number_of_crimes_l))),
    ))

    fig.update_layout(
        font=dict(size=20),
    )
    fig.update_layout(
        title={
            'text': 'Top 10 Neighbourhood per Number of Crimes Reported',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=25)
        }
    )
    st.plotly_chart(fig, use_container_width=True, )

    boa_reg_2018 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2018.shp')

    boa_reg_2019 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2019.shp')

    boa_reg_2020 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2020.shp')

    boa_reg_2021 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2021.shp')

    boa_reg_2022 = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/CC_2022.shp')

    breda_full_grid = gpd.read_file(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/Buas_BOAregistrion/Breda-Full_CBS-Grid.shp')


    def translate_name(crime):

        if crime == 'Fietsoverlast':

            return 'Bicycle Nuisance'

        elif crime == 'Parkeeroverlast':

            return 'Parking Nuisance'

        elif crime == 'Wildplassen':

            return 'Pee in public'

        elif crime == 'Afval':

            return 'Waste'

        elif crime == 'Jeugd':

            return 'Youth Nuisances'

        elif crime == 'Drugs/alcohol':

            return 'Drugs/Alcohol Nuisances'

        elif crime == 'Zwervers':

            return 'Homeless People'

        else:

            return ''


    def rename_column(df, column_name, new_name):

        df = df.rename(columns={column_name: new_name})

        return df


    boa_reg_2018.Soort = boa_reg_2018.Soort.apply(lambda x: translate_name(x))
    boa_reg_2018 = rename_column(boa_reg_2018, 'CBS_code', 'cbs_grid_code')

    boa_reg_2019.Soort = boa_reg_2019.Soort.apply(lambda x: translate_name(x))
    boa_reg_2019 = rename_column(boa_reg_2019, 'CBS_code', 'cbs_grid_code')

    boa_reg_2020.Soort = boa_reg_2020.Soort.apply(lambda x: translate_name(x))
    boa_reg_2020 = rename_column(boa_reg_2020, 'CBS_code', 'cbs_grid_code')

    boa_reg_2021.Soort = boa_reg_2021.Soort.apply(lambda x: translate_name(x))
    boa_reg_2021 = rename_column(boa_reg_2021, 'CBS_code', 'cbs_grid_code')

    boa_reg_2022.Soort = boa_reg_2022.Soort.apply(lambda x: translate_name(x))
    boa_reg_2022 = rename_column(boa_reg_2022, 'CBS_code', 'cbs_grid_code')

    oops = gpd.read_file('Municipality-AI-Powered-Web-App/finalApp/data/data/breda_grid_keys.gpkg')

    nusiances_2018 = pd.merge(oops, boa_reg_2018, on='cbs_grid_code', how='outer')
    nusiances_2019 = pd.merge(oops, boa_reg_2019, on='cbs_grid_code', how='outer')
    nusiances_2020 = pd.merge(oops, boa_reg_2020, on='cbs_grid_code', how='outer')
    nusiances_2021 = pd.merge(oops, boa_reg_2021, on='cbs_grid_code', how='outer')
    nusiances_2022 = pd.merge(oops, boa_reg_2022, on='cbs_grid_code', how='outer')


    def get_data_ready_for_map(df):
        df = df.rename(columns={'geometry_x': 'geometry'})
        df.crs = CRS.from_epsg(4326)
        df = df.to_crs(epsg=4326)
        df['geoid'] = df.index.astype('str')
        df = df[['cbs_grid_code', 'geoid', 'Count', 'geometry']]
        df = df.dropna(subset=['Count'])

        return df


    def clean_after_merge(df):

        df['Count_y'] = df['Count_y'].fillna(0)

        df['Count'] = df['Count_x'] + df['Count_y']

        df['geoid'] = df['geoid_x']

        df['geometry'] = df['geometry_x']

        df = df[['cbs_grid_code', 'geoid', 'geometry', 'Count']]

        return df


    nusiances_2018 = get_data_ready_for_map(nusiances_2018)
    nusiances_2019 = get_data_ready_for_map(nusiances_2019)
    nusiances_2020 = get_data_ready_for_map(nusiances_2020)
    nusiances_2021 = get_data_ready_for_map(nusiances_2021)
    nusiances_2022 = get_data_ready_for_map(nusiances_2022)

    nusiances_total = pd.merge(nusiances_2018, nusiances_2019, on='cbs_grid_code', how='outer')
    nusiances_total = clean_after_merge(nusiances_total)

    nusiances_total = pd.merge(nusiances_total, nusiances_2020, on='cbs_grid_code', how='outer')
    nusiances_total = clean_after_merge(nusiances_total)

    nusiances_total = pd.merge(nusiances_total, nusiances_2021, on='cbs_grid_code', how='outer')
    nusiances_total = clean_after_merge(nusiances_total)

    nusiances_total = pd.merge(nusiances_total, nusiances_2022, on='cbs_grid_code', how='outer')
    nusiances_total = clean_after_merge(nusiances_total)

    nusiances_total['geoid'] = nusiances_total.index.astype('str')
    nusiances_total = nusiances_total[['geoid', 'geometry', 'Count']]
    nusiances_total = nusiances_total.drop_duplicates(subset='geometry')

    nusiances_total['geoid'] = nusiances_total.index.astype('str')

    st.subheader("Mapping Nuisances in Breda:")
    st.markdown(
        "Explore the spatial distribution of registered nuisances from 2018 to 2022. Gain insights into evolving trends and patterns, offering a comprehensive view of the city's nuisance landscape over the years.")

    # interactive map

    nusiances_2018_json = nusiances_2018.__geo_interface__
    nusiances_2019_json = nusiances_2019.__geo_interface__
    nusiances_2020_json = nusiances_2020.__geo_interface__
    nusiances_2021_json = nusiances_2021.__geo_interface__
    nusiances_2022_json = nusiances_2022.__geo_interface__

    fig = go.Figure()


    def update_map(years):
        for year in years:
            if year == 2018:
                fig.add_trace(go.Choroplethmapbox(
                    geojson=nusiances_2018_json,
                    locations=nusiances_2018['geoid'],
                    z=nusiances_2018['Count'],
                    colorscale='YlOrRd',
                    zmin=0,
                    zmax=1000,
                    marker_opacity=0.3,
                    marker_line_width=0.2,
                    name='Nuisances 2018'
                ))
            elif year == 2019:
                fig.add_trace(go.Choroplethmapbox(
                    geojson=nusiances_2019_json,
                    locations=nusiances_2019['geoid'],
                    z=nusiances_2019['Count'],
                    colorscale='YlOrRd',
                    zmin=0,
                    zmax=1000,
                    marker_opacity=0.3,
                    marker_line_width=0.2,
                    name='Nuisances 2019'
                ))
            elif year == 2020:
                fig.add_trace(go.Choroplethmapbox(
                    geojson=nusiances_2020_json,
                    locations=nusiances_2020['geoid'],
                    z=nusiances_2020['Count'],
                    colorscale='YlOrRd',
                    zmin=0,
                    zmax=1000,
                    marker_opacity=0.3,
                    marker_line_width=0.2,
                    name='Nuisances 2020'
                ))
            elif year == 2021:
                fig.add_trace(go.Choroplethmapbox(
                    geojson=nusiances_2021_json,
                    locations=nusiances_2021['geoid'],
                    z=nusiances_2021['Count'],
                    colorscale='YlOrRd',
                    zmin=0,
                    zmax=1000,
                    marker_opacity=0.3,
                    marker_line_width=0.2,
                    name='Nuisances 2021'
                ))
            elif year == 2022:
                fig.add_trace(go.Choroplethmapbox(
                    geojson=nusiances_2022_json,
                    locations=nusiances_2022['geoid'],
                    z=nusiances_2022['Count'],
                    colorscale='YlOrRd',
                    zmin=0,
                    zmax=1000,
                    marker_opacity=0.3,
                    marker_line_width=0.2,
                    name='Nuisances 2022'
                ))


    available_years = [2018, 2019, 2020, 2021, 2022]

    selected_years = st.slider('Select years', min_value=min(available_years), max_value=max(available_years),
                               value=(min(available_years), max(available_years)))

    fig.data = []

    update_map(range(selected_years[0], selected_years[1] + 1))

    fig.update_layout(
        mapbox_style='carto-positron',
        mapbox_zoom=13,
        mapbox_center={'lat': 51.571915, 'lon': 4.768323},
    )

    fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

    colorbar_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers',
        marker=dict(
            colorscale='YlOrRd',
            cmin=0,
            cmax=1000,
            opacity=0.5,
            showscale=True,
            colorbar=dict(
                title='Nuisances Count',
                titleside='right',
                ticks='outside',
                tickmode='auto',
                ticklen=5,
                thickness=15
            )
        ),
        hoverinfo='none',
        showlegend=False
    )

    fig.add_trace(colorbar_trace)

    st.plotly_chart(fig)

    st.header("Neighbourhood Analysis")
    st.write(
        'Upon reviewing registrations from Breda, it becomes apparent that there is a significant variation in crime and nuisance rates among different neighborhoods. After identifying the ten neighborhoods with the highest crime rates in Breda, we further examined the occurrence of nuisances. We observed that regions experiencing high rates of crime also tend to have high levels of nuisances. This implies that an increase in the number of nuisances can sometimes lead to an uptick in crime. Additionally, we explored the correlation between the Quality of Life index and crime rate in each neighborhood. It was observed that neighborhoods with higher quality of life tend to be less susceptible to crime. Consequently, it is of utmost importance for law enforcement to assess the conditions in different neighborhoods, drawing on historical data, in order to allocate resources in a suitable and efficient manner.')

with tab3:
    crimes_reports_per_year_Breda = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/Misdrijven__soort_misdrijf__plaats_23052023_151236.csv',
        delimiter=';')

    keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    crimes_reports_per_year_Breda = crimes_reports_per_year_Breda.loc[keep]
    crimes_reports_per_year_Breda = crimes_reports_per_year_Breda.reset_index(drop=True)

    st.subheader('Crimes & Nuisances Analysis')

    recorded_crimes_Breda_per_month = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/Processed agony.csv',
                                                  dtype=dtype_mapping, low_memory=False)

    recorded_crimes_Breda_per_month = recorded_crimes_Breda_per_month.rename(
        columns={'Soort misdrijf': 'Type of Crime'})
    recorded_crimes_Breda_per_month = recorded_crimes_Breda_per_month.rename(columns={'Perioden': 'Periods'})
    recorded_crimes_Breda_per_month = recorded_crimes_Breda_per_month.rename(
        columns={'Wijken en buurten': 'Districts and neighbourhoods'})
    recorded_crimes_Breda_per_month = recorded_crimes_Breda_per_month.rename(
        columns={'Geregistreerde misdrijven (aantal)': 'Registered Crimes (total)'})

    ColumnsTokeep = ['Type of Crime', 'Periods', 'Districts and neighbourhoods', 'Registered Crimes (total)']
    TotalCrimes = recorded_crimes_Breda_per_month.drop(
        columns=[col for col in recorded_crimes_Breda_per_month.columns if col not in ColumnsTokeep])
    TotalcrimesBreda = TotalCrimes.drop(TotalCrimes.index[136:])

    TotalcrimesBreda['Registered Crimes (total)'] = TotalcrimesBreda['Registered Crimes (total)'].astype(float)


    def create_graph(df):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=df['Periods'], y=df['Registered Crimes (total)'], mode='lines', name='Registered Crimes'))

        x = np.arange(len(df))
        y = df['Registered Crimes (total)']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        trend_line = p(x)

        fig.add_trace(go.Scatter(x=df['Periods'], y=trend_line, mode='lines', name='Trend Line',
                                 line=dict(color='red', dash='dash')))

        fig.update_layout(
            title='Total Registered Crimes in Breda per month 2012 - 2023',
            xaxis_title='Months',
            yaxis_title='Number of Registered Crimes',
            legend_title='Legend'
        )

        return fig


    fig = create_graph(TotalcrimesBreda)

    st.plotly_chart(fig, use_container_width=True,
                    config={'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'resetScale2d', 'zoomIn2d', 'zoomOut2d']})

    st.write(
        "From the line plots above, we can see a downhill trend in the number of crimes. The number of registered crimes per month decreased by 600 since 2012. This shows that the current approach and development of technologies help police to prevent crimes and provide safety.")

    recorded_nuisances_Breda_per_year = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/Breda_nuisance.csv')

    recorded_nuisances_Breda_per_year = recorded_nuisances_Breda_per_year.drop(
        ['2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012'], axis=1)

    df = recorded_nuisances_Breda_per_year[
        recorded_nuisances_Breda_per_year['Overlast'] == 'Total nuisance registrations']
    df.drop(columns=["Regio's", "Overlast"], inplace=True)
    df = df.transpose()

    x = np.arange(len(df.index))
    y = df.iloc[:, 0]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    trend_line = p(x)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=y, mode='lines', name='Registered Nuisances'))

    fig.add_trace(
        go.Scatter(x=df.index, y=trend_line, mode='lines', name='Trend Line', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title='Total Registered Nuisances per month 2012 - 2022',
        xaxis_title='Month',
        yaxis_title='Number of Registered Nuisances',
        legend_title='Legend',
        font=dict(
            size=20
        ),
        # height =400,
        # width=100
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "While exploring data about the number of nuisances registered per month we expected the number of nuisances to decrease as the number of crimes decreased. In fact, the number of nuisances increased by 300 since 2012.")

    st.subheader("Influence of the season on crimes and nuisances")

    st.write(
        "After analysing both nuisances & crimes we can see that there are several spikes throughout each year of observation. We came up with a conclusion that these spikes are related to different seasons of the year and depending on the season of the year crimes probability can be higher or lower.")
    st.write(
        "After looking at this data we wanted to take a look at the correlation between the season of the year and the likelihood of crimes/nuisances happening")

    recorded_crimes_Breda_per_month_for_plot = pd.read_csv(
        'Municipality-AI-Powered-Web-App/finalApp/data/data/Processed agony.csv', delimiter=",", skiprows=4,
        dtype=dtype_mapping, low_memory=False)

    recorded_crimes_Breda_per_month_for_plot = recorded_crimes_Breda_per_month_for_plot.drop(
        ['3', 'misdrijven', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
         'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13',
         'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17',
         'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21',
         'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25',
         'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29',
         'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33',
         'Unnamed: 34', 'Unnamed: 35', 'Unnamed: 36', 'Unnamed: 37',
         'Unnamed: 38', 'Unnamed: 39', 'Unnamed: 40', 'Unnamed: 41',
         'Unnamed: 42', 'Unnamed: 43', 'Unnamed: 44', 'Unnamed: 45',
         'Unnamed: 46', 'Unnamed: 47', 'Unnamed: 48', 'Unnamed: 49',
         'Unnamed: 50', 'Unnamed: 51', 'Unnamed: 52', 'Unnamed: 53',
         'Unnamed: 54', 'Unnamed: 55', 'Unnamed: 56', 'Unnamed: 57',
         'Unnamed: 58', 'Unnamed: 59', 'Unnamed: 60', 'Unnamed: 61',
         'Unnamed: 62', 'Unnamed: 63'], axis=1)

    recorded_crimes_Breda_per_month_for_plot = recorded_crimes_Breda_per_month_for_plot[
        recorded_crimes_Breda_per_month_for_plot["Breda"] == "Breda"]

    recorded_crimes_Breda_per_month_for_plot = recorded_crimes_Breda_per_month_for_plot[
        recorded_crimes_Breda_per_month_for_plot["Totaal misdrijven"] == "Totaal misdrijven"]

    recorded_crimes_Breda_per_month_for_plot = recorded_crimes_Breda_per_month_for_plot.drop(
        ['Totaal misdrijven', 'Breda'], axis=1)

    recorded_crimes_Breda_per_month_for_plot.columns = ['month', 'number']


    def swap_columns(df, col1, col2):

        col_list = list(df.columns)
        x, y = col_list.index(col1), col_list.index(col2)
        col_list[y], col_list[x] = col_list[x], col_list[y]
        df = df[col_list]

        return df


    recorded_crimes_Breda_per_month_for_plot = swap_columns(recorded_crimes_Breda_per_month_for_plot, 'month', 'number')

    data = recorded_crimes_Breda_per_month_for_plot.iloc[:120]
    months = data.columns[2:]
    years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    color = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']
    year_num = 0

    fig = go.Figure()

    for i in [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]:
        x = list(range(1, 13))
        y = data.iloc[:, 0][i - 12:i]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(color=color[year_num]), name=years[year_num]))
        year_num += 1

    fig.update_layout(
        title='Total Number of Crimes Registered Over 12 Months per Year',
        xaxis_title='Month',
        yaxis_title='Number of Crimes Registered',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                      'November', 'December']
        ),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.5)'
        ),
        width=1200,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("Click on years to select or deselect them from the plot")

    import plotly.graph_objects as go

    data = df.iloc[:120]
    months = data.columns[2:]
    years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    color = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']
    year_num = 0

    fig = go.Figure()

    for i in [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]:
        x = list(range(1, 13))
        y = data.iloc[:, 0][i - 12:i]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(color=color[year_num]), name=years[year_num]))
        year_num += 1

    fig.update_layout(
        title='Total Number of Nuisances Registered Over 12 Months per Year',
        xaxis_title='Month',
        yaxis_title='Number of Nuisances Registered',
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                      'November', 'December']
        ),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            orientation='v'
        ),
        width=1200,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    st.header("Season Analysis")
    st.subheader("The graphs reveal distinct patterns in the rates of crime and nuisances throughout the year:")
    st.subheader("Nuisances:")
    st.markdown(
        "- Over the years, a consistent trend of increasing and decreasing nuisances can be observed during the summer and winter months respectively. January and February are notably safer compared to June and July, as they exhibit the highest occurrence of nuisances.")
    st.subheader("Crimes:")
    st.markdown(
        "- While it is challenging to establish a definite pattern for crimes, it is evident that March, August, and December are the least crime-prone months, whereas July, November, and February experience the highest crime rates.")
    st.subheader(
        "Considering our findings, it is evident that the season of the year significantly influences the rates of crime and nuisances. This information should be taken into account when determining the necessary resources to ensure city safety.")
