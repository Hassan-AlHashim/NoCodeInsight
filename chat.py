import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback
from dash.exceptions import PreventUpdate
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Assuming that OpenAI and necessary imports are correctly defined above this snippet
client = OpenAI(api_key = api_key)
model = "gpt-4-1106-preview"

def ask_openai(df, cleaned_df, cleaned_code, question, predictions_df=None, model_performance=None):
    # Construct the content message based on available data
    content_message = ""
    if df is not None:
        original_sample_data = df.head(10).to_string(index=False)
        content_message += f"Here is a sample of the original dataset:\n{original_sample_data}\n\n"

    if cleaned_df is not None and cleaned_code:
        cleaned_sample_data = cleaned_df.head(10).to_string(index=False)
        cleaning_code_message = f"The dataset has been cleaned using the following procedure:\n{cleaned_code}\n\n"
        content_message += f"Here is a sample of the cleaned dataset:\n{cleaned_sample_data}\n\n"
 
    # Add prediction results to the message if available
    if predictions_df is not None:
        predictions_sample_data = predictions_df.head(10).to_string(index=False)
        content_message += f"The model's predictions on the cleaned data are as follows:\n{predictions_sample_data}\n\n"
 
    # Add model performance metrics to the message if available
    if model_performance is not None:
        performance_str = "\n\nModel performance metrics:\n"
        performance_str += "\n".join([f"{key}: {value}" for key, value in model_performance.items()])
        content_message += performance_str
 
    # If no data is available at all, return a message indicating that
    if not content_message:
        return "No data available to display."
 
    # Prepare the messages for OpenAI
    messages = [
        {
            "role": "system",
            "content": "You are an AI that has access to a dataset and information about how it was cleaned. Answer questions based on the dataset and the cleaning process applied. Ensure providing actual insight based on the model and predictions including rmse value etc. Keep it short, simple, you are explaining to a beginner."
        },

        {
            "role": "user",
            "content": content_message
        },

        {
            "role": "user",
            "content": question
        }
    ]

 
    # Make a synchronous request to the OpenAI API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=300,
        temperature=0.3  # You can adjust the temperature based on desired creativity
    )

 
    # Accessing the message content
    ai_response = response.choices[0].message.content.strip()
    return ai_response


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