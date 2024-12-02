# with the env file
import os
from dotenv import load_dotenv
import pandas as pd
import pandas_gbq
from google.cloud import bigquery
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#TODO make another test set that is more raw/no assumptions and will be like the data streaming in

# Load environment variables
load_dotenv()

# Default configurable variables
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
BQ_QUERY = os.getenv("BQ_QUERY")
PRICE_LOWER_BOUND = int(os.getenv("PRICE_LOWER_BOUND", 1000))
PRICE_UPPER_BOUND = int(os.getenv("PRICE_UPPER_BOUND", 100000))
TOP_X_MODELS = int(os.getenv("TOP_X_MODELS", 30))
ODO_LOWER_BOUND = int(os.getenv("ODO_LOWER_BOUND", 1000))
ODO_UPPER_BOUND = int(os.getenv("ODO_UPPER_BOUND", 500000))
CAR_YEAR_THRESHOLD = int(os.getenv("CAR_YEAR_THRESHOLD", 2000))

def get_bq_df():
    """Fetch data from BigQuery."""
    client = bigquery.Client(project=BQ_PROJECT_ID)
    return pandas_gbq.read_gbq(BQ_QUERY, project_id=BQ_PROJECT_ID)

def filter_data(df):
    """Apply filtering and cleaning rules."""
    # Filtering by price bounds
    df = df[(df['price'] >= PRICE_LOWER_BOUND) & (df['price'] <= PRICE_UPPER_BOUND)]
    
    # Keeping only top X models
    top_models = df['model'].value_counts().nlargest(TOP_X_MODELS).index
    df = df[df['model'].isin(top_models)]
    
    # Filtering by odometer range
    df = df[(df['odometer'] >= ODO_LOWER_BOUND) & (df['odometer'] <= ODO_UPPER_BOUND)]
    
    # Filtering by cylinder count
    df = df[df['cylinder_count'].isin([4, 6, 8])] #maybe just have all of them?
    
    # Filtering by car year
    df = df[df['car_year'] >= CAR_YEAR_THRESHOLD] # maybe just have all the years?
    
    # Filtering by car condition
    df = df[df['car_condition'].isin(['good', 'excellent'])]
    
    # Filtering by transmission
    df = df[df['transmission'] == 'automatic']
    
    # Filtering by title status
    df = df[df['title_status'] == 'clean']
    
    # Filtering by fuel type
    df = df[df['fuel'] == 'gas']
    
    return df

def preprocess_for_ml(df):
    """Normalize and encode data for ML models."""
    # Map car condition to binary
    df['car_condition'] = df['car_condition'].map({'good': 0, 'excellent': 1})
    
    # Normalize odometer
    scaler = StandardScaler()
    df['odometer'] = scaler.fit_transform(df[['odometer']])
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['model', 'state'])
    
    # Replace invalid column names
    df.columns = df.columns.str.replace('-', '_')
    df.columns = df.columns.str.replace(' ', '')
    
    return df

def split_data(df):
    """Split data into train and test sets."""
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    return train_df, test_df

def save_to_csv(train_df, test_df, train_path="train_data.csv", test_path="test_data.csv"):
    """Save train and test datasets as local CSV files."""
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Train data saved to {train_path}")
    print(f"Test data saved to {test_path}")

if __name__ == "__main__":
    # Fetch data
    raw_df = get_bq_df()
    
    # Select necessary columns
    selected_columns = ['price', 'model', 'fuel', 'odometer', 'title_status', 
                        'transmission', 'cylinder_count', 'car_year', 
                        'car_condition', 'state']
    raw_df = raw_df[selected_columns]
    
    # Filter and clean data
    cleaned_df = filter_data(raw_df)
    
    # Drop unnecessary columns
    final_df = cleaned_df.drop(columns=['fuel', 'title_status', 'transmission'])
    
    # Preprocess for ML
    ml_ready_df = preprocess_for_ml(final_df)
    
    # Split data
    train_df, test_df = split_data(ml_ready_df)
    
    # Save data locally
    save_to_csv(train_df, test_df)
