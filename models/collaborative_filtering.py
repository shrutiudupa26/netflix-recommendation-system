import os
import dotenv
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F
from itertools import product

class ALSCollaborativeFiltering:
    def __init__(self):
        """
        Initialize the ALS Collaborative Filtering System with config from .env file
        """
        # Load environment variables
        dotenv.load_dotenv()
        
        # Configuration from environment
        self.config = {
            "app_name": os.getenv("APP_NAME", "Netflix ALS Collaborative Filtering"),
            "master": os.getenv("SPARK_MASTER", "local"),
            "executor_memory": os.getenv("EXECUTOR_MEMORY", "3g"),
            "driver_memory": os.getenv("DRIVER_MEMORY", "2g"),
            "spark_temp_dir": os.getenv("SPARK_TEMP_DIR", ""),
            "data_path": os.getenv("DATA_PATH", "./data/cleaned/netflix_ratings_features_combined.parquet"),
            "als_cold_start": os.getenv("ALS_COLD_START", "drop"),
            "train_ratio": float(os.getenv("TRAIN_RATIO", "0.8")),
            "seed": int(os.getenv("RANDOM_SEED", "42")),
            "log_level": os.getenv("LOG_LEVEL", "ERROR"),
            "load_saved_model": os.getenv("LOAD_SAVED_MODEL", "false").lower() == "true",
            # AWS S3 configuration
            "aws_s3_path": os.getenv("AWS_S3_PATH_COLLAB_MODEL", ""),
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "")
        }
            
        # Initialize Spark session
        self.spark = self._initialize_spark()
        self.model = None
        self.df = None
        self.train = None
        self.test = None
        self.movie_titles_df = None
        
    def _initialize_spark(self):
        """
        Initialize and configure Spark session with S3 capabilities
        """
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

        # Set the log level for the SparkContext
        spark.sparkContext.setLogLevel(self.config["log_level"])
        spark.sparkContext.setCheckpointDir(f"{self.config['spark_temp_dir']}/spark-checkpoints")

        return spark
        
    def load_data(self):
        """
        Load and preprocess Netflix combined dataset containing both ratings and titles
        """
        self.df = self.spark.read.parquet(self.config["data_path"])

        self.df = self.df.withColumnRenamed("movie_id", "movieId") \
                         .withColumnRenamed("customer_id", "userId")

        print(f"Number of rows in dataset: {self.df.count()}")

        self.train, self.test = self.df.randomSplit(
            [self.config["train_ratio"], 1 - self.config["train_ratio"]],
            seed=self.config["seed"]
        )
        return self.df
    
    def find_best_model(self, train_data, validation_data, ranks, regParams, maxIters):
        """
        Perform search over ALS hyperparameters to find the best model.
        """
        min_rmse = float('inf')
        best_model = None
        best_params = {}
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        # Search over all combinations
        for rank, reg, maxIter in product(ranks, regParams, maxIters):
            als = ALS(
                maxIter=maxIter,
                rank=rank,
                regParam=reg,
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                coldStartStrategy="drop",
                nonnegative=True
            )
            als.setCheckpointInterval(2)
            model = als.fit(train_data)
            predictions = model.transform(validation_data)
            rmse = evaluator.evaluate(predictions)
            print(f"rank={rank}, regParam={reg}, maxIter={maxIter} => RMSE={rmse:.4f}")
            if rmse < min_rmse:
                min_rmse = rmse
                best_model = model
                best_params = {"rank": rank, "regParam": reg, "maxIter": maxIter}

        print(f"\nBest Model Parameters: {best_params}, RMSE={min_rmse:.4f}")
        return best_model, best_params

    
    def save_model(self):
        """
        Save the model to AWS S3
        """
        self.model.write().overwrite().save(self.config["aws_s3_path"])
        print(f"ALS model saved to S3: {self.config['aws_s3_path']}")
        return self.config["aws_s3_path"]
    
    def load_model(self):
        """
        Load a previously saved model from AWS S3
        """
        self.model = ALSModel.load(self.config["aws_s3_path"])
        print(f"ALS model loaded from S3: {self.config['aws_s3_path']}")
        return self.model

    def generate_recommendations(self, top_n=5, user_sample_size=5):
        """
        Generate movie recommendations for a sample of users using titles from the same dataset
        """
        total_users = self.train.select("userId").distinct().count()
        print(f"Total distinct users: {total_users}")

        users = self.train.select("userId").distinct().orderBy(F.rand(seed=42)).limit(user_sample_size)
        print(f"Sampled {users.count()} users")

        recommendations = self.model.recommendForUserSubset(users, top_n)

        exploded_recs = recommendations.select("userId", F.explode("recommendations").alias("rec")) \
                                       .select("userId", F.col("rec.movieId").alias("movieId"))

        movie_title_df = self.df.select("movieId", "title").dropna(subset=["title"]).distinct()

        recommendations_with_titles = exploded_recs.join(
            movie_title_df,
            on="movieId",
            how="left"
        ).select("userId", "title")

        user_recommendations = recommendations_with_titles.groupBy("userId").agg(
            F.collect_list("title").alias("recommended_movies")
        )

        return user_recommendations



# Main block to execute
if __name__ == "__main__":
    als_cf = ALSCollaborativeFiltering()
    try:
        # Load data
        als_cf.load_data()
        # Load model if available
        if als_cf.config["load_saved_model"]:
            als_cf.load_model()
        if not als_cf.model:
            # Find best model
            ranks = [10, 20]
            regParams = [0.05]
            maxIters = [10]
            #sampled_train = als_cf.train.sample(withReplacement=False, fraction=0.0001, seed=42)
            #sampled_val = als_cf.test.sample(withReplacement=False, fraction=0.0001, seed=42)
            best_model, best_params = als_cf.find_best_model(als_cf.train, als_cf.test, ranks, regParams, maxIters)
            als_cf.model = best_model
            als_cf.save_model()
            print("Best model trained and saved with best params", best_params)

        # Generate recommendations
        recommendations = als_cf.generate_recommendations(top_n=5, user_sample_size=5)
        print("\nMovie Recommendations for Sample Users:")
        recommendations.show(truncate=False)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        als_cf.spark.stop()  
