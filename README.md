# NoCodeInsight
This repository hosts a Python application demonstrating how to integrate and utilize the OpenAI API, in creating a a streamlined approach for predictive analysis. The motivation behind this project is to bridge the gap between ML/AI analysis and non STEM backgrounds to bring ML capabilities to non STEM or coding professionals.

## Getting Started
These instructions will guide you through the process of setting up and running this project on your local machine for development and testing purposes.

## Files: 
1. app.py: Script containing front end for GUI using DASH
2. clean_train.py: Script containing functions for automated cleaning and training of data and models
3. chat.py: Script to integrate OpenAI Chat interface with model and cleaned data

### Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.11
- Pip (Python package manager)

### Installation
Follow these simple steps to get your development environment running:

1. **Clone the Repository**

   Clone the repository to your local machine using this command: git clone

2. **Installing Dependencies**

   Install the required packages using the following command: pip install -r requirements.txt

3. **Set up OpenAI API Key**

   - Create OpenAI account, and get API Key from the following link: https://platform.openai.com/api-keys
   - Add OpenAI API key to .env file in the following form "OPENAI_API_KEY = your_openai_api_key:
  
4. **Running Application**

   To run the application, run the following command in the terminal: python app.py, this command will start an application on a webserver.

### Usage
Once application is running, user is able to upload a data set, the following are the steps to complete the analysis:

1. Upload and clean data set
2. Train data set: Users have two options for training. The buttont "train" utilizes simple regression models, suitable for linear data sets, while the "advanced training" button includes random forest models, suitable for non-linear data sets. User may use both models to compare results
3. Predictions buttons are included for both advanced, and linear models
4. Once analysis is complete, user is able to look at figures, and chat with OpenAI to discuss results

This code was developed as part of the 1.125: Arch & Engineering Software Systems class at MIt for the Fall 2023 semester instructed by Professor John Williams, and Dr. abel Sanches. For further inquiries or feedback please contact Hassan Alhashim (hwhashim@mit.edu) or Odai Elyas (oaelyas2@mit.edu)
   
   

   
