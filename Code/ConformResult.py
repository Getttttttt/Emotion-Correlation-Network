import pandas as pd
import os
import glob
import re

# Pattern to split the filename and extract correlation type, emotion, and distance
pattern = r'(\w+)_(\w+)_distance_(\d+)\.csv'

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Iterate over each file in the directory matching the pattern for CSV files
for file in glob.glob('./result/data/*.csv'):
    # Extract filename details
    filename = os.path.basename(file)
    match = re.search(pattern, filename)
    if match:
        correlation_type, emotion, distance = match.groups()
        
        # Read the current file
        data = pd.read_csv(file)
        
        # Add columns for the metadata extracted from the filename
        data['correlation_type'] = correlation_type
        data['emotion'] = emotion
        data['distance'] = int(distance)
        
        # Append to the combined DataFrame
        combined_data = pd.concat([combined_data, data], ignore_index=True)

# Display the combined DataFrame structure and the first few rows to confirm
combined_data.head(), combined_data.columns

combined_data.to_csv('./Output/Result.csv')