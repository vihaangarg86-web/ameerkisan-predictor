import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import joblib
import itertools
import logging
import warnings

# Suppress informational logging from cmdstanpy
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)

CLEAN_DATA_FILE = 'maize_clean_for_training.csv'
MODEL_FILE = 'maize_model.joblib'

def train_model():
    print("--- Starting SUPER-ADVANCED model training and tuning ---")

    # 1. Load your clean data
    try:
        df = pd.read_csv(CLEAN_DATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: Clean data file '{CLEAN_DATA_FILE}' not found. Run clean.py first.")
        return

    df['ds'] = pd.to_datetime(df['ds'])
    print(f"Data loaded. Training on {len(df)} data points.")
    
    # Check if we have enough data for a 2-year initial train
    if len(df) < 800: # Approx 2 years + buffer
        initial_train_days = f'{int(len(df) * 0.7)} days' # Use 70% of data
        print(f"Warning: Less than 2 years of data. Setting initial train to {initial_train_days}")
    else:
        initial_train_days = '730 days'

    # 2. Define the NEW "search grid" for fine-tuning
    # We are adding more flexibility with changepoint_range and weekly_seasonality
    param_grid = {
        'changepoint_prior_scale': [0.1, 0.5, 1.0],  # More flexible
        'seasonality_prior_scale': [10.0, 15.0],     # Test seasonality strength
        'changepoint_range': [0.8, 0.9, 0.95],      # NEW: Look at 80%, 90%, or 95% of the data
        'weekly_seasonality': [True, False]         # NEW: Check for weekly patterns
    }

    # Generate all possible parameter combinations
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    rmses = []  # We'll store the error (RMSE) for each model here
    
    print(f"Starting hyperparameter tuning... testing {len(all_params)} model combinations.")
    print(f"This may take 5-10 minutes... please be patient.")

    # 3. Cross-Validation Loop
    for i, params in enumerate(all_params):
        print(f"  [Test {i+1}/{len(all_params)}] Testing params: {params}")
        
        # Initialize the model with the new parameters
        model_params = {
            'yearly_seasonality': True,
            'daily_seasonality': False,
        }
        model_params.update(params) # This will add the params from the grid
        model = Prophet(**model_params)
        
        model.fit(df)

        # Run cross-validation
        try:
            df_cv = cross_validation(
                model, 
                initial=initial_train_days, 
                period='180 days', 
                horizon='30 days', 
                parallel="processes",
                disable_diagnostics=True
            )
            
            df_p = performance_metrics(df_cv, metrics=['rmse'])
            rmses.append(df_p['rmse'].values[0])
            print(f"    -> Avg. Error (RMSE): {df_p['rmse'].values[0]:.4f}")
            
        except Exception as e:
            print(f"    -> WARNING: Cross-validation failed for params {params}. Error: {e}")
            rmses.append(float('inf')) # Give this a very high error so it's not chosen

    # 4. Find the Best Parameters
    if not all(rmse == float('inf') for rmse in rmses):
        best_params_index = rmses.index(min(rmses))
        best_params = all_params[best_params_index]
        best_rmse = min(rmses)
        
        print("\n--- Tuning Complete ---")
        print(f"Lowest RMSE (avg. error) found: {best_rmse:.4f}")
        print(f"Best Parameters found: {best_params}")
    else:
        print("\n--- Tuning Failed ---")
        print("Cross-validation failed for all parameters. Using default settings.")
        best_params = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0, 'changepoint_range': 0.8, 'weekly_seasonality': False}

    # 5. Train the FINAL model
    print("\nTraining final model with the best parameters...")
    
    final_model_params = {
        'yearly_seasonality': True,
        'daily_seasonality': False,
    }
    final_model_params.update(best_params)
    final_model = Prophet(**final_model_params)
    
    final_model.fit(df)
    
    print("Final model trained.")

    # 6. Save the final model
    joblib.dump(final_model, MODEL_FILE)

    print(f"---")
    print(f"SUCCESS: New, hyper-tuned model saved as '{MODEL_FILE}'")
    print(f"---")

if __name__ == "__main__":
    train_model()