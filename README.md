# Netflix Dataset EDA and Recommendation System Preparation

## Overview
This project performs exploratory data analysis (EDA) and data preparation for building a Netflix recommendation system. The workflow includes:
1. Loading and cleaning Netflix movie data
2. Integrating IMDb movie details
3. Analyzing user ratings and movie features patterns
4. Preparing final datasets for recommendation modeling

*Note: The model building part will be updated soon.*

## Datasets Required
To run this analysis, you'll need the following datasets:

1. **Netflix Movie Titles**: [Download from Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
   - File: `movie_titles.csv`
   - Contains Netflix movie IDs, titles, and release years

2. **Netflix User Ratings**: [Download from Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
   - Files: `combined_data_1.txt` through `combined_data_4.txt`
   - Contains user rating data

3. **IMDb Movie Details**: [Download from Kaggle](https://www.kaggle.com/datasets/stephanerappeneau/350-000-movies-from-themoviedborg)
   - File: `AllMoviesDetailsCleaned.csv`
   - Contains comprehensive movie metadata from IMDb

## Setup

### 1. Install python requirements using the `requirements.txt` file to install required libraries:

```bash
pip install -r requirements.txt
```

### 2. Create a .env file in your project root with these variables: 

ANTHROPIC_API_KEY=your_api_key_here
CLAUDE_MODEL=your_claude_model
INPUT_TOKEN_LIMIT=8000 
OUTPUT_TOKEN_LIMIT=100
MAX_REQUESTS_PER_MINUTE=50

To get an Anthropic API key:
Sign up at Anthropic's website and create an API key in your account settings.
Add it to the .env file.
*Note: The optimal values for INPUT_TOKEN_LIMIT, OUTPUT_TOKEN_LIMIT, and MAX_REQUESTS_PER_MINUTE will depend on your specific Anthropic model version and API subscription tier.*

### 3. Your file structure should look like:
netflix-recommendation/
├── data/
│ ├── raw/
│ │ ├── movie_titles.csv
│ │ ├── combined_data_1.txt
│ │ ├── combined_data_2.txt
│ │ ├── combined_data_3.txt
│ │ ├── combined_data_4.txt
│ │ └── AllMoviesDetailsCleaned.csv
│ └── cleaned/
├── notebooks/
│ └── Netflix_EDA.ipynb
├── .env
└── requirements.txt

 ```bash
   # Create the directory structure for datasets
   mkdir -p data/raw data/cleaned
```
Before running the notebook, place all datasets in the `data/raw/` directory. The cleaned datasets after EDA will be stored in `data/cleaned/`
   
   # Download datasets manually (see links above) into data/raw/
