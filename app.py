import dash
import base64
import io
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, callback
from dash import dcc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
import pandas as pd
import traceback
import pickle
import plotly.express as px
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import plotly.express as px

import logging

logging.basicConfig(level=logging.INFO)
    
from clean_train import suggest_cleaning_procedure, suggest_model_training_code, suggest_advanced_model_training_code
from chat import ask_openai, update_openai_response


# Define the dark theme from the Bootstrap template
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc.themes.CYBORG]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Layout of the Dash app
app.layout = dbc.Container(
    [
        html.H4(
            "No Code Insight",
            className="bg-primary text-white p-2 mb-2 text-center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Data Cleaning & Visualization", className='text-center text-primary mb-4', style={'font-size': '24px'}),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'color': 'dark'
                            },
                            multiple=False
                        ),
                        html.Button('Visualize Data', id='visualize-button', n_clicks=0, className='btn btn-primary btn-block my-2', style={'borderRadius': '5px'}),
                        html.Button('Clean Data', id='clean-data-button', n_clicks=0, className='btn btn-warning btn-block my-2', style={'borderRadius': '5px'}),
                        dcc.Store(id='store-cleaned-data'),
                        dcc.Store(id='store-cleaning-code'),
                        dcc.Store(id='store-original-data'),
                        dcc.Store(id='store-prediction-df'),
                        dcc.Store(id='store-prediction-advanced-df'),
                        dcc.Dropdown(id='column-selector', options=[], placeholder="Select a column", className='my-2', style={'borderRadius': '5px'}),
                        html.Button('Train Model', id='train-model-button', n_clicks=0, className='btn btn-success btn-block my-2', style={'borderRadius': '5px'}),
                        html.Div(id='model-training-output', className='my-2'),
                        html.Button('Train Advanced Model', id='train-advanced-model-button', n_clicks=0, className='btn btn-info btn-block my-2', style={'borderRadius': '5px'}),
                        html.Div(id='advanced-model-training-output', className='my-2'),
                        html.H4("Upload Data for Prediction", className='text-center text-primary mt-4 mb-2', style={'font-size': '24px'}),
                       dcc.Upload(
                            id='upload-data-predict',
                            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'margin': '10px',
                                'color': 'dark'
                            },
                            multiple=False
                        ),
                        html.Button('Predict', id='predict-button', n_clicks=0, className='btn btn-danger btn-block my-2', style={'borderRadius': '5px'}),
                        html.Button('Predict Using Advanced Model', id='predict-advanced-button', n_clicks=0, className='btn btn-info btn-block my-2', style={'borderRadius': '5px'}),
                        html.Div(id='prediction-output', className='my-2'),
                        html.Div(id='prediction-advanced-output', className='my-2'),
                        dcc.Input(id='openai-question', type='text', placeholder='Ask a question to OpenAI...', className='form-control my-2', style={'borderRadius': '5px'}),
                        html.Button('Ask OpenAI', id='ask-openai-button', n_clicks=0, className='btn btn-secondary btn-block my-2', style={'borderRadius': '5px'}),
                        html.Div(id='openai-response', className='p-3 border bg-light my-2', style={'borderRadius': '5px'}),
                    ],
                    width=4,
                    style={'background-color': '#14171b', 'borderRadius': '15px'}
                ),
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(
                                    html.Div(id='data-view'),
                                    label="Grid",
                                ),
                                dbc.Tab(
                                    html.Div(id='graph-container'),
                                    label="Conventional Model Performance",
                                ),
                                dbc.Tab(
                                    html.Div(id='graph-advanced-container'),
                                    label="Advanced Model Performance",
                                ),
                                dbc.Tab(
                                    html.Div(id='feature-importance-container'),  # Changed to html.Div
                                    label="Feature Importance",
                                ),
                                dbc.Tab(
                                    html.Div(id='oob-error-container'),
                                    label="Metrics",
                                ),
                                dbc.Tab(
                                    html.Div(id='decision-path-container'),
                                    label="Decision Path",
                                ),
                            ],
                        ),
                    ],
                    width=8,
                    style={'padding': '20px'}
                ),
            ],
            className="g-4",
        )
    ],
    fluid=True,
    className='py-5'
)

# Helper function to parse uploaded data

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(["This file type is not supported."])
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])
    return df

 

# Helper function to create a data table

def create_data_table(df):
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },

        style_cell={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white',
            'border': '1px solid grey'
        },

        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
        page_action="native",
        page_current= 0,
        page_size= 10,
    )


# Define the function to plot feature importance
def plot_feature_importance(model, feature_names):

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

    else:
        return "Feature importances are not available for this model type."

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h')
    fig.update_layout(
        plot_bgcolor='rgb(0, 0, 0)',
        paper_bgcolor='rgb(0, 0, 0)',
        font_color='white',
        xaxis=dict(gridcolor='rgb(80, 80, 80)'),
        yaxis=dict(gridcolor='rgb(80, 80, 80)')
    )

    return fig


 
@app.callback(
    [
        Output('data-view', 'children'),
        Output('store-cleaned-data', 'data'),
        Output('store-cleaning-code', 'data'),
        Output('store-original-data', 'data')
    ],
    [Input('visualize-button', 'n_clicks'),
     Input('clean-data-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('column-selector', 'value')]
)

def combined_callback(visualize_clicks, clean_clicks, contents, filename, selected_column):
    # Determine which input was triggered
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
 
    # If no data is uploaded, do nothing
    if not contents:
        raise PreventUpdate
 
    df = parse_data(contents, filename)

    if df is None:
        return html.Div("There was an error processing this file."), None, None, None
 
    # Initialize the variables for storing data
    stored_cleaned_data = None
    stored_cleaning_code = None
    stored_original_data = None
 
    # Check which button was clicked
    if button_id == 'visualize-button' and visualize_clicks:
        stored_original_data = df.to_json(date_format='iso', orient='split')
        return create_data_table(df), None, None, stored_original_data

    elif button_id == 'clean-data-button' and clean_clicks:
        # Ensure the target column is provided before cleaning
        if not selected_column:
            return "Please select a target column before cleaning.", None, None, None
        
        cleaning_function_code = suggest_cleaning_procedure(df, selected_column)  # Pass the target column
        logging.info(f"Cleaning function code: {cleaning_function_code}")
        exec(cleaning_function_code, globals())

        cleaned_df = clean_dataframe(df)
        stored_cleaned_data = cleaned_df.to_json(date_format='iso', orient='split')
        stored_cleaning_code = cleaning_function_code
        stored_original_data = df.to_json(date_format='iso', orient='split')  # Store original data
        return create_data_table(cleaned_df), stored_cleaned_data, stored_cleaning_code, stored_original_data

    # If neither button has been clicked, we only need to update the message
    return html.Div("Upload a file to see options here."), None, None, None

@app.callback(
    Output('column-selector', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

def update_column_selector(contents, filename):
    if contents is None:
        raise PreventUpdate

    df = parse_data(contents, filename)
    if df is None:
        return []
    # Here we create the options for the dropdown based on DataFrame columns
    return [{'label': col, 'value': col} for col in df.columns]
 

# Callback to switch between DataFrame and Plot tabs
@app.callback(
    Output("content", "children"),
    Input("tabs", "active_tab")
)

def switch_tab(at):
    if at == "tab-data":
        return html.Div(id='output-data-upload')

    elif at == "tab-plot":
        return dcc.Graph(id='prediction-plot')

    return html.P("This shouldn't happen!")

@app.callback(
    Output('model-training-output', 'children'),
    [Input('train-model-button', 'n_clicks')],
    [State('store-cleaned-data', 'data'), State('store-original-data', 'data'), State('column-selector', 'value')]
)

def train_model_callback(n_clicks, cleaned_data_json, original_data_json, target_column):

    if n_clicks is None or n_clicks < 1 or not target_column:
        raise PreventUpdate
    
    cleaned_df = pd.read_json(cleaned_data_json, orient='split') if cleaned_data_json else None
    original_df = pd.read_json(original_data_json, orient='split') if original_data_json else None

    if cleaned_df is None or original_df is None:
        return "No cleaned or original data available for training."

    # Get the suggested training code from the OpenAI API
    training_code = suggest_model_training_code(cleaned_df, original_df, target_column)
    logging.info(f"Training code: {training_code}")

    # Execute the training code
    try:
        # Define a local namespace to execute the code
        local_namespace = {
            'pd': pd,
            'train_test_split': train_test_split,
            'LinearRegression': LinearRegression,
            'LogisticRegression': LogisticRegression,  
            'MinMaxScaler': MinMaxScaler,  # Ensure MinMaxScaler is imported
            'cleaned_data': cleaned_df,
            'original_data': original_df,
            'target_column': target_column
        }

        # Execute the received training code to define the function
        exec(training_code, local_namespace)

        # Now call the function 'train_predictive_model' with the required arguments
        exec("predictive_model, scaler, X_train, X_test, y_train, y_test = train_predictive_model(cleaned_data, original_data, target_column)", local_namespace)

        # Extract the model, scaler, and data splits from the local_namespace
        model = local_namespace.get('predictive_model')
        scaler = local_namespace.get('scaler')
        X_train = local_namespace.get('X_train')
        X_test = local_namespace.get('X_test')
        y_train = local_namespace.get('y_train')
        y_test = local_namespace.get('y_test')

        # Save the model and scaler
        with open('predictive_model.pkl', 'wb') as file:
            pickle.dump(model, file)

        with open('scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        return "Model trained successfully and saved."

    except Exception as e:
        # Print the full traceback to help debug the issue
        traceback.print_exc()

        return f"An error occurred during model training: {e}"


@app.callback(
    Output('advanced-model-training-output', 'children'),
    [Input('train-advanced-model-button', 'n_clicks')],
    [State('store-cleaned-data', 'data'),
     State('store-original-data', 'data'),
     State('column-selector', 'value')]
)

def train_advanced_model_callback(n_clicks, cleaned_data_json, original_data_json, target_column):

    if n_clicks is None or n_clicks < 1 or not target_column:
        raise PreventUpdate

    cleaned_df = pd.read_json(cleaned_data_json, orient='split') if cleaned_data_json else None
    original_df = pd.read_json(original_data_json, orient='split') if original_data_json else None

    if cleaned_df is None or original_df is None:
        return "No cleaned or original data available for training."

    training_code = suggest_advanced_model_training_code(cleaned_df, original_df, target_column)

    # Debugging: Print the training code to verify its correctness
    logging.info(f"Training code: {training_code}")

    try:
        local_namespace = {
            'pd': pd,
            'train_test_split': train_test_split,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'RandomForestClassifier': RandomForestClassifier,
            'MinMaxScaler': MinMaxScaler,
            'cleaned_data': cleaned_df,
            'original_data': original_df,
            'target_column': target_column
        }

        exec(training_code, local_namespace)

        # Assuming the function name inside training_code is 'train_advanced_model'
        exec("advanced_model, scaler, X_train, X_test, y_train, y_test = train_advanced_model(cleaned_data, original_data, target_column)", local_namespace)

        model = local_namespace.get('advanced_model')
        scaler = local_namespace.get('scaler')
        X_train = local_namespace.get('X_train')
        X_test = local_namespace.get('X_test')
        y_train = local_namespace.get('y_train')
        y_test = local_namespace.get('y_test')
 
        # Save the model and scaler
        with open('advanced_model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)

        with open('advanced_scaler.pkl', 'wb') as scaler_file:  # Corrected the file name for the scaler
            pickle.dump(scaler, scaler_file)

        return "Advanced model trained successfully and saved."

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred during advanced model training: {e}"
 
# Update the predict callback
@app.callback(
    [
        Output('prediction-output', 'children'),
        Output('store-prediction-df', 'data'),
        Output('graph-container', 'children')
    ],
    [Input('predict-button', 'n_clicks')],
    [
        State('upload-data-predict', 'contents'),
        State('upload-data-predict', 'filename'),
        State('store-cleaned-data', 'data'),
        State('store-cleaning-code', 'data'),
        State('column-selector', 'value')
    ]
)

 
def predict(n_clicks, contents, filename, cleaned_data_json, cleaning_code, target_column):
    if n_clicks is None or n_clicks < 1:
        return html.Div(), None, None

    if contents is None:
        return "No prediction data uploaded.", None, None

    original_prediction_df = parse_data(contents, filename)

    if original_prediction_df is None:
        return "Error in parsing prediction data.", None, None

    exec(cleaning_code, globals())
    cleaned_prediction_df = clean_dataframe(original_prediction_df)

    with open('predictive_model.pkl', 'rb') as file:
        model = pickle.load(file)

    with open('target_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)


    # Drop the target column for prediction, if it exists in the DataFrame
    features_for_prediction = cleaned_prediction_df.drop(target_column, axis=1, errors='ignore')

    # Predict using the model
    predictions_normalized = model.predict(features_for_prediction) 

    # Reshape the predictions if necessary before applying inverse scaling
    if predictions_normalized.ndim == 1:
        predictions_normalized = predictions_normalized.reshape(-1, 1)

    predictions_rescaled = scaler.inverse_transform(predictions_normalized).flatten()
    original_prediction_df['Predictions'] = predictions_rescaled

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(original_prediction_df[target_column], original_prediction_df['Predictions']))
 

    fig = px.scatter(
        x=original_prediction_df[target_column],
        y=original_prediction_df['Predictions'],
        labels={'x': 'Actual', 'y': 'Predicted'},
        #title=f'Actual vs Predicted Plot (RMSE: {rmse:.2f})'
   )
    fig.add_scatter(x=[original_prediction_df[target_column].min(), original_prediction_df[target_column].max()],
                    y=[original_prediction_df[target_column].min(), original_prediction_df[target_column].max()],
                    mode='lines', showlegend=False)

    fig.update_layout(
        plot_bgcolor='rgb(0, 0, 0)',
        paper_bgcolor='rgb(0, 0, 0)',
        font_color='white',
        xaxis=dict(gridcolor='rgb(80, 80, 80)'),
        yaxis=dict(gridcolor='rgb(80, 80, 80)')
    )


    predictions_json = original_prediction_df.to_json(date_format='iso', orient='split')
    return create_data_table(original_prediction_df), predictions_json, dcc.Graph(figure=fig)

@app.callback(
    [
        Output('prediction-advanced-output', 'children'),
        Output('store-prediction-advanced-df', 'data'),
        Output('graph-advanced-container', 'children'),
        Output('feature-importance-container', 'children'),  # Update to target the container
        Output('decision-path-container', 'children'),
        Output('oob-error-container', 'children')
    ],

    [Input('predict-advanced-button', 'n_clicks')],
    [
        State('upload-data-predict', 'contents'),
        State('upload-data-predict', 'filename'),
        State('store-cleaned-data', 'data'),
        State('store-cleaning-code', 'data'),
        State('column-selector', 'value')
    ]
)

def predict_advanced(n_clicks, contents, filename, cleaned_data_json, cleaning_code, target_column):
    if n_clicks is None or n_clicks < 1:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if contents is None:
        return "No prediction data uploaded.", None, None, None, None, None
 

    original_prediction_df = parse_data(contents, filename)
    if original_prediction_df is None:
        return "Error in parsing prediction data.", None, None, None, None, None
 

    exec(cleaning_code, globals())
    cleaned_prediction_df = clean_dataframe(original_prediction_df)
 
    with open('advanced_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('advanced_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    features_for_prediction = cleaned_prediction_df.drop(target_column, axis=1, errors='ignore')
    predictions_normalized = model.predict(features_for_prediction)
    if predictions_normalized.ndim == 1:
        predictions_normalized = predictions_normalized.reshape(-1, 1)
    predictions_rescaled = scaler.inverse_transform(predictions_normalized).flatten()
    original_prediction_df['Predictions'] = predictions_rescaled
    rmse = sqrt(mean_squared_error(original_prediction_df[target_column], original_prediction_df['Predictions']))

    # Actual vs Predicted Plot
    actual_vs_predicted_fig = create_actual_vs_predicted_plot(original_prediction_df, target_column)

    # Feature Importance Plot
    feature_importance_fig = create_feature_importance_plot(model, features_for_prediction)

    # Decision Path Plot
    decision_path_fig = create_decision_path_subplots(model, features_for_prediction)

 
    # OOB Error Estimate
    oob_error_text = create_oob_error_estimate(model)
    MAPE = np.mean(np.abs((original_prediction_df[target_column] - original_prediction_df['Predictions']) / original_prediction_df[target_column])) * 100
    RMSE = np.sqrt(np.mean((original_prediction_df[target_column] - original_prediction_df['Predictions']) ** 2))

    evalutation = f"MAPE: {MAPE:.2f}%, RMSE: {RMSE:.2f}, OOB Error Estimate: {oob_error_text}"

    predictions_json = original_prediction_df.to_json(date_format='iso', orient='split')
    return create_data_table(original_prediction_df), predictions_json, dcc.Graph(figure=actual_vs_predicted_fig), dcc.Graph(figure=feature_importance_fig), dcc.Graph(figure=decision_path_fig), html.Div(evalutation)

 
 

# Implementations for plots and error estimate
def create_actual_vs_predicted_plot(df, target_column):
    fig = px.scatter(
        df, x=target_column, y='Predictions',
        labels={'x': 'Actual', 'y': 'Predicted'},
        title='Actual vs Predicted Values'
    )
    fig.add_scatter(x=[df[target_column].min(), df[target_column].max()],
                    y=[df[target_column].min(), df[target_column].max()],
                    mode='lines', showlegend=False)
    fig.update_layout(plot_bgcolor='rgb(0, 0, 0)', paper_bgcolor='rgb(0, 0, 0)', font_color='white')
    return fig

 

def create_feature_importance_plot(model, features_df):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df = pd.DataFrame({'Feature': features_df.columns, 'Importance': importances})
        fig = px.bar(df.sort_values(by='Importance', ascending=False), x='Importance', y='Feature', orientation='h')
        fig.update_layout(plot_bgcolor='rgb(0, 0, 0)', paper_bgcolor='rgb(0, 0, 0)', font_color='white')
        return fig
    return go.Figure() 


def create_decision_path_subplots(model, features_df):
    if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):

        # Create a subplot layout with 2 rows and 3 columns
        fig = sp.make_subplots(rows=2, cols=3, subplot_titles=[f'Tree {i+1}' for i in range(6)])

        # Loop through the first 6 trees
        for i in range(6):
            row = i // 3 + 1
            col = i % 3 + 1
            tree = model.estimators_[i]
            decision_path = tree.decision_path(features_df).toarray()

            # Create a heatmap for each tree's decision path
            heatmap = px.imshow(
                decision_path,
                color_continuous_scale='Blues',
                aspect='auto',
                labels={'color': "Decision Path Value"}
            )

            # Modify layout to remove color axis for individual plots
            heatmap.update_layout(coloraxis_showscale=False)

            # Add the heatmap to the subplot
            for trace in heatmap.data:
                fig.add_trace(trace, row=row, col=col)

        # Update layout to match the dark theme
        fig.update_layout(
            plot_bgcolor='rgb(0, 0, 0)',
            paper_bgcolor='rgb(0, 0, 0)',
            font_color='white',
            title_text="Decision Paths of First 6 Trees",
            height=800,
            width=1200
        )
        return fig

    return go.Figure()  # Return empty figure if not a RandomForest

def create_oob_error_estimate(model):
    if hasattr(model, 'oob_score_'):
        oob_error = 1 - model.oob_score_
        return f"Out-of-Bag Error Estimate: {oob_error:.4f}"
    return "OOB error estimate not available. Ensure the model was trained with `oob_score=True`."
 
@app.callback(
    Output('openai-response', 'children'),
    [Input('ask-openai-button', 'n_clicks')],
    [State('openai-question', 'value'),
     State('store-cleaned-data', 'data'),
     State('store-cleaning-code', 'data'),
     State('store-original-data', 'data'),
     State('store-prediction-df', 'data')] 
)

def update_openai_response(n_clicks, question, stored_cleaned_data, stored_cleaning_code, stored_original_data, stored_predictions_data):

    if n_clicks is None or n_clicks < 1 or not question:
        raise PreventUpdate

    # Load the data from the stored JSON if available
    cleaned_df = pd.read_json(stored_cleaned_data, orient='split') if stored_cleaned_data else None
    df = pd.read_json(stored_original_data, orient='split') if stored_original_data else None
    predictions_df = pd.read_json(stored_predictions_data, orient='split') if stored_predictions_data else None

    cleaned_code = stored_cleaning_code if stored_cleaning_code else None

    # Call the ask_openai function with all the available data
    openai_response = ask_openai(df, cleaned_df, cleaned_code, question, predictions_df)
    return html.Div(openai_response)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)