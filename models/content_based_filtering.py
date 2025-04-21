import os
import dotenv
import pickle
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
import traceback

# Define this function at module level
def split_genres(text):
    return text.split("|")

class ContentBasedRecommender:
    def __init__(self):
        """Initialize Content-Based Recommender with config from .env"""
        dotenv.load_dotenv()
        
        self.config = {
            "app_name": os.getenv("APP_NAME", "Netflix Content-Based Recommender"),
            "master": os.getenv("SPARK_MASTER", "local[*]"),
            "executor_memory": os.getenv("EXECUTOR_MEMORY", "4g"),
            "driver_memory": os.getenv("DRIVER_MEMORY", "4g"),
            "spark_temp_dir": os.getenv("SPARK_TEMP_DIR", "/tmp"),
            "data_path": os.getenv("DATA_PATH", "./data/cleaned/netflix_ratings_features_combined.parquet"),
            "seed": int(os.getenv("RANDOM_SEED", "42")),
            "log_level": os.getenv("LOG_LEVEL", "ERROR"),
            "load_saved_model": os.getenv("LOAD_SAVED_MODEL", "false").lower() == "true",
            # AWS S3 configuration
            "aws_s3_path": os.getenv("AWS_S3_PATH_CONTENT_MODEL", ""),
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "")
        }
        
        # Feature weights - updated based on feature importance analysis
        self.weights = {
            "genre": 0.3,
            "overview": 0.3,
            "vote_avg": 0.2,
            "vote_count": 0.1,
            "year": 0.1
        }
            
        # Initialize Spark session
        self.spark = self._initialize_spark()
        self.df = None
        self.similarity_matrix = None
        
        # Initialize feature count tracking variables
        self.num_genre_features = 0
        self.num_overview_features = 0
        self.num_vote_features = 2
        self.num_year_features = 0
        
    def _initialize_spark(self):
        """Initialize and configure Spark session with S3 capabilities"""
        spark = SparkSession.builder \
            .appName(self.config["app_name"]) \
            .master(self.config["master"]) \
            .config("spark.executor.memory", self.config["executor_memory"]) \
            .config("spark.driver.memory", self.config["driver_memory"]) \
            .config("spark.local.dir", self.config["spark_temp_dir"]) \
            .config("spark.jars.packages", 
                    "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.367") \
            .config("spark.hadoop.fs.s3a.access.key", self.config["aws_access_key_id"]) \
            .config("spark.hadoop.fs.s3a.secret.key", self.config["aws_secret_access_key"]) \
            .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
                    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
            .config("spark.sql.broadcastTimeout", "600") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.default.parallelism", "12") \
            .getOrCreate()

        spark.sparkContext.setLogLevel(self.config["log_level"])
        return spark
        
    def load_data(self):
        """Load and preprocess Netflix dataset"""
        self.df = self.spark.read.parquet(self.config["data_path"])

        # Extract unique movies with features
        self.movies_df = self.df.select(
            col("movie_id").cast("int").alias("movie_id"),  
            col("title").cast("string").alias("title"),
            col("overview").cast("string").alias("overview"),    
            col("genres").cast("string").alias("genres"),   
            col("vote_average").cast("float").alias("vote_average"),
            col("vote_count").cast("float").alias("vote_count"),
            col("release_year_bins").cast("string").alias("release_year_bins")
        ).dropDuplicates(["movie_id"])
        
        # Handle missing overviews by replacing with empty string
        self.movies_df = self.movies_df.fillna({"overview": ""})

    def preprocess_features(self):
        """Preprocess features for similarity calculation"""
        # Collect as Python objects first 
        movies_list = self.movies_df.collect()
        
        # Create pandas DataFrame from collected data
        movies_pd = pd.DataFrame([row.asDict() for row in movies_list])

        # TF-IDF Vectorizer for genres - using the module-level function instead of lambda
        tfidf_genres = TfidfVectorizer(tokenizer=split_genres)
        genre_tfidf = tfidf_genres.fit_transform(movies_pd["genres"])
        self.num_genre_features = genre_tfidf.shape[1]
        
        # TF-IDF Vectorizer for overview text - limit to max 30 features
        tfidf_overview = TfidfVectorizer(max_features=30, stop_words='english')
        overview_tfidf = tfidf_overview.fit_transform(movies_pd["overview"])
        self.num_overview_features = overview_tfidf.shape[1]

        # Log-transform vote count
        movies_pd["vote_count_log"] = np.log1p(movies_pd["vote_count"])

        # Scale vote_average and vote_count_log
        scaler = MinMaxScaler()
        vote_scaled = scaler.fit_transform(movies_pd[["vote_average", "vote_count_log"]])

        # One-hot encode release_year_bins
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        release_year_encoded = encoder.fit_transform(movies_pd[["release_year_bins"]])
        self.num_year_features = release_year_encoded.shape[1]

        # Combine all features into a single NumPy array
        final_features = np.hstack([
            genre_tfidf.toarray(),
            overview_tfidf.toarray(),
            vote_scaled,
            release_year_encoded
        ])

        # Create a copy of only movie_id and title
        movies_processed = movies_pd[["movie_id", "title"]].copy()
        
        # Put processed features into the new DataFrame
        feature_columns = [f"feat_{i}" for i in range(final_features.shape[1])]
        for i, col in enumerate(feature_columns):
            movies_processed[col] = final_features[:, i]
        
        # Assign to class attribute
        self.movies_df_processed = movies_processed
        
        # Save pre-processing models for future use
        self.preprocessing_models = {
            'tfidf_genres': tfidf_genres,
            'tfidf_overview': tfidf_overview,
            'scaler': scaler,
            'encoder': encoder
        }
    
        return self.movies_df_processed
    
    def compute_similarity(self, weights=None):
        """Compute cosine similarity between items using weighted features"""
        # Use provided weights or defaults
        current_weights = weights if weights else self.weights
            
        # Get feature columns (all columns except movie_id and title)
        feature_cols = [col for col in self.movies_df_processed.columns 
                        if col not in ["movie_id", "title"]]
        
        # Calculate feature boundaries
        genre_end = self.num_genre_features
        overview_end = genre_end + self.num_overview_features
        vote_end = overview_end + self.num_vote_features
        
        # Determine feature groups based on indices
        genre_features = [f"feat_{i}" for i in range(genre_end)]
        overview_features = [f"feat_{i}" for i in range(genre_end, overview_end)]
        vote_avg_feature = [f"feat_{overview_end}"]
        vote_count_feature = [f"feat_{overview_end + 1}"]
        year_features = [f"feat_{i}" for i in range(vote_end, len(feature_cols))]
        
        # Extract feature vectors
        genre_vecs = self.movies_df_processed[genre_features].values
        overview_vecs = self.movies_df_processed[overview_features].values
        vote_avg_vecs = self.movies_df_processed[vote_avg_feature].values
        vote_count_vecs = self.movies_df_processed[vote_count_feature].values
        year_vecs = self.movies_df_processed[year_features].values
        
        # Calculate similarity matrices for each feature group
        genre_sim = cosine_similarity(genre_vecs)
        overview_sim = cosine_similarity(overview_vecs)
        vote_avg_sim = cosine_similarity(vote_avg_vecs)
        vote_count_sim = cosine_similarity(vote_count_vecs)
        year_sim = cosine_similarity(year_vecs) 
        
        # Combine with weights
        similarity_matrix = (
            current_weights["genre"] * genre_sim +
            current_weights["overview"] * overview_sim +
            current_weights["vote_avg"] * vote_avg_sim +
            current_weights["vote_count"] * vote_count_sim +
            current_weights["year"] * year_sim
        )
            
        # Store the similarity matrix as an instance variable
        self.similarity_matrix = similarity_matrix
        return similarity_matrix
    
    def recommend_similar_items(self, title, top_n=10, similarity_matrix=None):
        """Recommend similar items based on a given title"""    
        idx = self.title_to_idx[title]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1] 
        recommendations = [
            (self.idx_to_title[idx], self.idx_to_movieid[idx], score) 
            for idx, score in sim_scores
        ]
        return recommendations
    
    def recommend_for_user(self, user_id, top_n=10, similarity_matrix=None):
        """Generate recommendations for a user based on their watching history"""
        # Get user's watched movies
        user_movies = self.df.filter(col("customer_id") == user_id) \
                            .select("title") \
                            .distinct() \
                            .collect()
        
        user_titles = [row.title for row in user_movies]
        # Get indices of user movies
        user_indices = [self.title_to_idx[title] for title in user_titles if title in self.title_to_idx]
        # Calculate average similarity
        user_sim = np.mean([self.similarity_matrix[idx] for idx in user_indices], axis=0)
        # Create scores list
        movie_scores = list(enumerate(user_sim))
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        # Filter out movies the user has already seen
        movie_scores = [(idx, score) for idx, score in movie_scores 
                        if self.idx_to_title[idx] not in user_titles]
        # Get top N recommendations
        top_recommendations = movie_scores[:top_n]
        recommendations = [
            (self.idx_to_title[idx], self.idx_to_movieid[idx], score)
            for idx, score in top_recommendations
        ]
        return recommendations

    def set_feature_weights(self, weights):
        """Set custom weights for different features"""
        self.weights.update(weights)
        # Recalculate similarity if already computed
        if self.similarity_matrix is not None:
            self.compute_similarity()
        return self.weights
    
    def evaluate_for_users(self, num_users=100, top_n=5):
        """Evaluate recommendations for random users with optimized weights"""
        # Get random user IDs
        user_sample = self.df.select("customer_id").distinct() \
                             .orderBy(rand(seed=self.config["seed"])) \
                             .limit(num_users) \
                             .collect()
        user_ids = [row.customer_id for row in user_sample]
        # Generate single-feature weight configurations
        feature_weights = {
            "genre_only": {"genre": 1.0, "overview": 0.0, "vote_avg": 0.0, "vote_count": 0.0, "year": 0.0},
            "overview_only": {"genre": 0.0, "overview": 1.0, "vote_avg": 0.0, "vote_count": 0.0, "year": 0.0},
            "vote_avg_only": {"genre": 0.0, "overview": 0.0, "vote_avg": 1.0, "vote_count": 0.0, "year": 0.0},
            "vote_count_only": {"genre": 0.0, "overview": 0.0, "vote_avg": 0.0, "vote_count": 1.0, "year": 0.0},
            "year_only": {"genre": 0.0, "overview": 0.0, "vote_avg": 0.0, "vote_count": 0.0, "year": 1.0}
        }
        
        # Store user agreement scores
        user_agreements = {feature: [] for feature in feature_weights.keys()}
        # For each feature, evaluate recommendations for each user
        for feature_name, weights in feature_weights.items():
            # Compute similarity matrix once for this feature
            similarity_matrix = self.compute_similarity(weights)
            # For each user, get recommendations and measure agreement
            for user_id in user_ids:
                # Get recommendations for this user
                recommendations = self.recommend_for_user(
                    user_id, 
                    top_n=top_n,
                    similarity_matrix=similarity_matrix
                )
                if recommendations:
                    # Calculate average similarity score for this user's recommendations
                    avg_score = sum(score for _, _, score in recommendations) / len(recommendations)
                    user_agreements[feature_name].append(avg_score)
        # Calculate average agreement across users for each feature
        feature_avg_agreements = {
            feature: sum(scores) / len(scores) if scores else 0.0
            for feature, scores in user_agreements.items()
        }
        return feature_avg_agreements
    
    def set_optimal_weights_from_analysis(self, feature_agreements):
        """Set optimal weights based on feature importance analysis"""
        # Normalize agreement scores to sum to 1.0
        total = sum(feature_agreements.values())
        
        # Calculate weights proportional to agreement scores
        weights = {
            "genre": feature_agreements.get("genre_only", 0) / total,
            "overview": feature_agreements.get("overview_only", 0) / total,
            "vote_avg": feature_agreements.get("vote_avg_only", 0) / total,
            "vote_count": feature_agreements.get("vote_count_only", 0) / total,
            "year": feature_agreements.get("year_only", 0) / total
        }
        
        # Update recommender weights
        self.set_feature_weights(weights)
        return weights
    
    def save_model(self):
        """Save the trained recommender model to AWS S3"""
        # Create model data to save
        model_data = {
            'weights': self.weights,
            'movies_df_processed': self.movies_df_processed,
            'similarity_matrix': self.similarity_matrix,
            'preprocessing_models': self.preprocessing_models,
            'idx_to_title': self.idx_to_title,
            'title_to_idx': self.title_to_idx,
            'idx_to_movieid': self.idx_to_movieid,
            'num_genre_features': self.num_genre_features,
            'num_overview_features': self.num_overview_features,
            'num_vote_features': self.num_vote_features,
            'num_year_features': self.num_year_features
        }
        
        # Extract bucket and key
        s3_path = self.config["aws_s3_path"]
        bucket_key = s3_path[5:]  # Remove 's3://'
        parts = bucket_key.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else "content_model.pkl"
    
        # Upload to S3
        s3 = boto3.client(
            's3',
            aws_access_key_id=self.config["aws_access_key_id"],
            aws_secret_access_key=self.config["aws_secret_access_key"]
        )
        s3.put_object(Bucket=bucket, Key=key, Body=pickle.dumps(model_data))
        print(f"Model saved to S3: s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"
    
    def load_model(self, config=None):
        """Load model from S3 using boto3"""
        # Clean and validate S3 path
        s3_path = self.config["aws_s3_path"]
        # Extract bucket and key
        bucket_key = s3_path[5:]  # Remove 's3://'
        parts = bucket_key.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else "content_model.pkl"

        # Download from S3
        s3 = boto3.client(
            's3',
            aws_access_key_id=self.config["aws_access_key_id"],
            aws_secret_access_key=self.config["aws_secret_access_key"]
        )
        response = s3.get_object(Bucket=bucket, Key=key)
        model_data = pickle.loads(response['Body'].read())
        print(f"Model loaded from S3: s3://{bucket}/{key}")
            
        # Load data from the model
        recommender.weights = model_data['weights']
        recommender.movies_df_processed = model_data['movies_df_processed']
        recommender.similarity_matrix = model_data['similarity_matrix']
        recommender.preprocessing_models = model_data['preprocessing_models']
        recommender.idx_to_title = model_data['idx_to_title']
        recommender.title_to_idx = model_data['title_to_idx'] 
        recommender.idx_to_movieid = model_data['idx_to_movieid']
        recommender.num_genre_features = model_data['num_genre_features']
        recommender.num_overview_features = model_data['num_overview_features']
        recommender.num_vote_features = model_data['num_vote_features']
        recommender.num_year_features = model_data['num_year_features']
        return recommender

if __name__ == "__main__":  
    # Create and train recommender
    recommender = ContentBasedRecommender()
    try:
        # Try to load a saved model first if configured
        if recommender.config["load_saved_model"]:
            loaded_recommender = recommender.load_model()
            if loaded_recommender.similarity_matrix is not None:
                print("Successfully loaded saved model")
            else:
                print("No saved model found or load failed. Training new model...")
                
        # If no model loaded, train a new one
        if recommender.similarity_matrix is None:
            # Load and preprocess data
            recommender.load_data()
            recommender.preprocess_features()
            
            # Initialize lookup dictionaries
            recommender.idx_to_title = dict(zip(
                range(len(recommender.movies_df_processed)), 
                recommender.movies_df_processed["title"]
            ))
            recommender.title_to_idx = {v: k for k, v in recommender.idx_to_title.items()}
            recommender.idx_to_movieid = dict(zip(
                range(len(recommender.movies_df_processed)), 
                recommender.movies_df_processed["movie_id"]
            ))
            
            # Optional: Analyze and optimize weights
            user_evaluation = recommender.evaluate_for_users(num_users=50, top_n=5)
            best_weights = recommender.set_optimal_weights_from_analysis(user_evaluation)
            
            # Save the trained model
            model_path = recommender.save_model()
            print(f"Model saved to {model_path}")
        
        # Test recommendations
        all_titles = list(recommender.title_to_idx.keys())
        sample_movie = random.choice(all_titles)
        print(f"Recommendations for '{sample_movie}':")
        recommendations = recommender.recommend_similar_items(sample_movie, top_n=5)
        for title, movie_id, score in recommendations:
            print(f"- {title} (ID: {movie_id}), Similarity: {score:.4f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        recommender.spark.stop()