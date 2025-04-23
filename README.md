# Netflix Recommendation System

## Overview
This project builds a comprehensive Netflix recommendation system using both collaborative filtering and content-based approaches. The workflow includes:
1. Loading and cleaning Netflix movie data
2. Integrating IMDb movie details
3. Analyzing user ratings and movie features patterns
4. Implementing recommendation models:
   - Collaborative filtering using Alternating Least Squares (ALS)
   - Content-based recommendations using cosine similarity

For deeper insights into my approach and thought process, feel free to check out my medium posts on the [EDA process](https://medium.com/@udupashruti/first-try-at-building-an-end-to-end-recommendation-system-exploratory-data-analysis-c90cfd1b6ad6) and [Model Building process](https://medium.com/@udupashruti/first-try-at-building-a-recommendation-system-model-development-c82ef6e9e775):

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

## Project Structure

```
netflix-recommendation/
├── data/
│   ├── raw/
│   │   ├── movie_titles.csv
│   │   ├── combined_data_1.txt
│   │   ├── combined_data_2.txt
│   │   ├── combined_data_3.txt
│   │   ├── combined_data_4.txt
│   │   └── AllMoviesDetailsCleaned.csv
│   └── cleaned/
│       └── netflix_ratings_features_combined.parquet
├── notebooks/
│   └── Netflix_EDA.ipynb
├── models/
│   ├── collaborative_filtering.py
│   └── content_based_filtering.py
├── .env
└── requirements.txt
```

## Setup

### 1. Install Java and Spark
Apache Spark requires Java 8 or later. 
Install JDK from https://www.oracle.com/java/technologies/downloads/. 
Install Spark from https://spark.apache.org/downloads.html.

### 2. Install Python Requirements
Install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in your project root with the following variables:

```
# Anthropic API Configuration
ANTHROPIC_API_KEY=your_api_key_here
CLAUDE_MODEL=your_claude_model
INPUT_TOKEN_LIMIT=8000 
OUTPUT_TOKEN_LIMIT=100
MAX_REQUESTS_PER_MINUTE=50

# Spark Configuration
APP_NAME=Netflix Recommendation System
SPARK_MASTER=local[*]
EXECUTOR_MEMORY=4g
DRIVER_MEMORY=4g
SPARK_TEMP_DIR=/tmp
DATA_PATH=./data/cleaned/netflix_ratings_features_combined.parquet
RANDOM_SEED=42
LOG_LEVEL=ERROR
LOAD_SAVED_MODEL=false
TRAIN_RATIO=0.8
ALS_COLD_START=drop

# AWS S3 Configuration 
AWS_S3_PATH_COLLAB_MODEL=s3://your-bucket/als_model
AWS_S3_PATH_CONTENT_MODEL=s3://your-bucket/content_model
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

*Note:*
- To get an Anthropic API key:
   - Sign up at Anthropic's website and create an API key in your account settings
   - Add it to the .env file
   - The optimal values for INPUT_TOKEN_LIMIT, OUTPUT_TOKEN_LIMIT, and MAX_REQUESTS_PER_MINUTE will depend on your specific Anthropic model version and API subscription tier
- To set up AWS access for S3 storage:
   - Create an AWS account if you don't have one
   - Navigate to IAM (Identity and Access Management) in the AWS Console and create a new user with access and required policy.
   - Generate access key and secret key
   - Add these credentials to your .env file
   - Create an S3 bucket to store your models and update the S3 paths in the .env file accordingly

### 4. Directory Setup
Create the necessary directory structure:

```bash
mkdir -p data/raw data/cleaned models notebooks
```

Download datasets manually (see links above) into the `data/raw/` directory.
The cleaned datasets after EDA will be stored in the `data/cleaned/` directory.

## Project Components

### 1. Data Preparation and EDA
Run the `notebooks/Netflix_EDA.ipynb` Jupyter notebook to:
- Clean and transform the raw Netflix data
- Integrate IMDb movie features
- Analyze user rating patterns
- Create the combined dataset for modeling
- Save the processed data to `data/cleaned/netflix_ratings_features_combined.parquet`

### 2. Collaborative Filtering Model
The `models/collaborative_filtering.py` script implements:
- ALS matrix factorization for user-item recommendations
- Hyperparameter tuning for optimal model performance
- Model evaluation on test data
- Generation of personalized movie recommendations

To run the collaborative filtering model:
```bash
python models/collaborative_filtering.py
```

### 3. Content-Based Recommender
The `models/content_based_filtering.py` script implements:
- Feature extraction from movie metadata (genres, overview, vote average etc)
- TF-IDF vectorization for text features
- Weighted cosine similarity calculations
- Feature importance analysis and weight optimization
- Movie-to-movie and user-to-movie recommendations

To run the content-based recommender:
```bash
python models/content_based_filtering.py
```

*Note:* Both models support saving and loading from AWS S3. Therefore, set `LOAD_SAVED_MODEL=true` to load previously saved models.



