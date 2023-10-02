import streamlit as st
import pandas as pd
from prophet import Prophet
import geopandas as gpd
import datetime
import plotly.express as px
import webbrowser

st.header("AI for Safety")

map_c, rest_c = st.columns([1.5, 0.5])

predictions_dictionary = {}

predictons_created = False

df = pd.read_csv('Municipality-AI-Powered-Web-App/finalApp/data/data/data_for_model.csv')


def get_data_for_model(f_path='Municipality-AI-Powered-Web-App/finalApp/data/data/data_for_model.csv'):
    df = pd.read_csv(f_path, parse_dates=True).drop(columns=['Unnamed: 0'])
    df.date = pd.to_datetime(df.date)
    df.number_of_crimes = df.number_of_crimes.replace('       .', '       0')
    df.number_of_crimes = df.number_of_crimes.apply(lambda x: int(x[6:]))

    return df


df_for_predictions = get_data_for_model()


def get_dictionary_for_model(f_path='Municipality-AI-Powered-Web-App/finalApp/data/data/data_for_model.csv'):
    df = get_data_for_model(f_path)

    name_date_crime_nhood_dfs = {

        n_name: pd.DataFrame({
            'ds': [date for date in
                   df[df.neighbourhood_name == n_name].drop(columns='neighbourhood_name').rename(columns={
                       'date': 'ds',
                       'number_of_crimes': 'y'
                   }).ds.unique()],
            'y': [sum(df[df.neighbourhood_name == n_name].drop(columns='neighbourhood_name').rename(columns={
                'date': 'ds',
                'number_of_crimes': 'y'
            })[df[df.neighbourhood_name == n_name].drop(columns='neighbourhood_name').rename(columns={
                'date': 'ds',
                'number_of_crimes': 'y'
            }).ds == date]['y']) for date in
                  df[df.neighbourhood_name == n_name].drop(columns='neighbourhood_name').rename(columns={
                      'date': 'ds',
                      'number_of_crimes': 'y'
                  }).ds.unique()]
        }) for n_name in df.neighbourhood_name.unique()

    }

    return name_date_crime_nhood_dfs


crimes_dictionary = get_dictionary_for_model()


def create_models_for_neighbourhoods(crimes_dictionary,
                                     f_path='Municipality-AI-Powered-Web-App/finalApp/data/data/data_for_model.csv'):
    df = get_data_for_model(f_path)

    models_per_n = {

        n_name: Prophet(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01,
                        seasonality_mode='multiplicative').fit(crimes_dictionary[n_name]) for n_name in
        df.neighbourhood_name.unique()

    }

    return models_per_n


models_dict = create_models_for_neighbourhoods(crimes_dictionary)


def get_predictions(future_dates, models_dict, all_neighbourhoods=True, spec_neighbourhoods=[],
                    f_path='Municipality-AI-Powered-Web-App/finalApp/data/data/data_for_model.csv'):
    if all_neighbourhoods:

        df = get_data_for_model(f_path)

        pred = {
            name: {
                'dates': future_dates['ds'][0],
                'predicted_number_of_crimes': [models_dict[name].predict(future_dates).loc[0, 'yhat']]
            }

            for name in df.neighbourhood_name.unique()
        }

        return pred

    else:

        df = get_data_for_model(f_path)

        pred = {
            name: {
                'dates': future_dates['ds'][0],
                'predicted_number_of_crimes': [models_dict[name].predict(future_dates).loc[0, 'yhat']]
            }

            for name in df.neighbourhood_name.unique()
        }

        return pred


tab1, tab2, tab3 = st.tabs(["Crimes Forecast", "Data Entry", "Documentation"])

with tab1:
    tab1.markdown(

        '''Here you can get a monthly forecast of number of crimes in different neighbourhoods in Breda.'''
    )

    col1, col2 = st.columns(2)

    with col1:

        predictions = {}

        d = st.date_input(
            "Pick a month you want to forecast (Pick 1st day of the month)!",
            datetime.date(2023, 6, 22))

        options = st.multiselect(
            'Filter by neighbourhoods',
            df.neighbourhood_name.unique())

        if st.button('Get a Forecast!'):

            if options and predictons_created:
                predict_df = pd.DataFrame(predictions_dictionary)
                predict_df = predict_df[predict_df.name.isin(options)]
                predict_df = predict_df.sort_values(by=['predicted_number_of_crimes'])

                fig = px.bar(predict_df, x='name', y='predicted_number_of_crimes')

                st.plotly_chart(fig, use_container_width=True)

            if d:

                if options and options != []:

                    with col2:

                        date = pd.DataFrame({'ds': pd.date_range(start=d, periods=1, freq='MS')})

                        predictions = get_predictions(date, models_dict)

                        predictions_dictionary = {
                            'name': [name for name, prediction in predictions.items()],
                            'predicted_number_of_crimes': [prediction['predicted_number_of_crimes'][0] for
                                                           name, prediction in predictions.items()]
                        }

                        predict_df = pd.DataFrame(predictions_dictionary)
                        predict_df = predict_df[predict_df.name.isin(options)]
                        predict_df = predict_df.sort_values(by=['predicted_number_of_crimes'])

                        fig = px.bar(predict_df, x='name', y='predicted_number_of_crimes')

                        st.plotly_chart(fig, use_container_width=True)

                        predictons_created = True



                else:

                    with col2:

                        date = pd.DataFrame({'ds': pd.date_range(start=d, periods=1, freq='MS')})

                        predictions = get_predictions(date, models_dict)

                        predictions_dictionary = {
                            'name': [name for name, prediction in predictions.items()],
                            'predicted_number_of_crimes': [prediction['predicted_number_of_crimes'][0] for
                                                           name, prediction in predictions.items()]
                        }

                        predict_df = pd.DataFrame(predictions_dictionary)
                        predict_df = predict_df.sort_values(by=['predicted_number_of_crimes'])

                        fig = px.bar(predict_df, x='name', y='predicted_number_of_crimes')

                        st.plotly_chart(fig, use_container_width=True)

                        predictons_created = True

    if predictons_created:
        st.dataframe(predict_df)

with tab2:
    tab2.markdown(

        '''Here you can enter new data about number of crimes regestired in different neighbourhoods. This will help to improve the model which will result in more accurate predictions.'''
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        option = st.selectbox(
            "Choose the neighbourhood",
            (df.neighbourhood_name.unique())
        )

    with col2:
        d = st.date_input(
            "Set the date for the crime report",
            datetime.date(2023, 6, 22))

    with col3:
        number = st.number_input('Insert the number of crimes registered', step=1)

    if st.button('Upload new data!'):
        st.write('Thank you for providing more data and improving the model!')
        st.write("Let's make Breda safer together!")

with tab3:
    tab3.markdown(
        '''
        Here you can find all the documnetation / reports we created while working on the project.
        '''
    )

    if st.button('Data Quality Report'):
        webbrowser.open(
            'https://edubuas-my.sharepoint.com/:b:/g/personal/223834_buas_nl/EQMoBQDK9XdMpMObu-o_FrsBjeV6Xgv1CMoUJZ5YZSghqQ?e=aOLdUE')

    if st.button('Ethics Report'):
        webbrowser.open(
            'https://edubuas-my.sharepoint.com/:w:/g/personal/224215_buas_nl/EXKE-ZuIFKBBjqh94BRRm8IBCr51o5L75S73NvEtztcTsA?e=ezPE9X')
