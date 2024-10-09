import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import r2_score, accuracy_score
import pickle
import folium

# Step 1: Load the Data from CSV
df = pd.read_csv("filtered_dataset_7km.csv")

# Step 2: Data Preprocessing
df['Hour'] = pd.to_datetime(df['Hour'], format='%H').dt.strftime('%H')

# Step 3: Downsample the Data (optional)
# df_sampled = df.sample(frac=0.1, random_state=42)

# Step 4: Clustering and Training Prediction Model
def train_prediction_model():
    coords = df[['StartLat', 'StartLon']].values
    agglomerative = AgglomerativeClustering(n_clusters=70).fit(coords)
    df['Cluster'] = agglomerative.labels_

    # Calculate the number of people in each cluster per hour
    cluster_counts = df.groupby(['Cluster', 'Hour']).size().reset_index(name='NumPeople')

    # Merge the counts back into the original dataframe
    df_with_counts = df.merge(cluster_counts, on=['Cluster', 'Hour'], how='left')

    # Features for the prediction: Latitude, Longitude, Hour
    X = df_with_counts[['StartLat', 'StartLon', 'Hour']].copy()
    X['Hour'] = pd.to_numeric(X['Hour'])

    # Target: Number of people in the cluster
    y = df_with_counts['NumPeople']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a prediction model (RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model using MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # R² Score
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2}")

    # Rounded accuracy (exact match)
    y_pred_rounded = np.round(y_pred)
    y_test_rounded = np.round(y_test)
    rounded_accuracy = accuracy_score(y_test_rounded, y_pred_rounded)
    print(f"Rounded Accuracy (exact match): {rounded_accuracy}")

    # Accuracy within ±1 person
    threshold_accuracy = within_threshold(y_test, y_pred, threshold=1)
    print(f"Accuracy within ±1 person: {threshold_accuracy}")

    # Save the trained model to a pickle file
    with open('trained_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved as 'trained_model.pkl'")

    return model

# Helper function to calculate accuracy within a threshold
def within_threshold(y_true, y_pred, threshold=1):
    return np.mean(np.abs(y_true - y_pred) <= threshold)

# Train the model and save it
trained_model = train_prediction_model()

# Step 5: Prediction and Map Visualization

# Function to predict the number of people based on latitude, longitude, and hour
def predict_num_people(lat, lon, hour):
    input_data = pd.DataFrame({
        'StartLat': [lat],
        'StartLon': [lon],
        'Hour': [int(hour)]
    })

    prediction = trained_model.predict(input_data)
    print(f"Predicted number of people at location ({lat}, {lon}) at hour {hour}: {int(prediction[0])}")
    return int(prediction[0])

# Function to plot the predicted cluster and nearby clusters on a map using folium
def plot_predicted_and_nearby_clusters(lat, lon, hour, num_nearby_clusters=3):
    # Filter data for the specified hour
    filtered_df = df[df['Hour'] == hour]

    if filtered_df.empty:
        print(f"No data available for the hour {hour}")
        return

    # Get the number of samples in the filtered dataset
    num_samples = len(filtered_df)

    # Set the number of clusters based on the number of available samples
    n_clusters = min(70, num_samples)

    # Perform clustering (AgglomerativeClustering)
    coords = filtered_df[['StartLat', 'StartLon']].values
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters).fit(coords)
    filtered_df['Cluster'] = agglomerative.labels_

    # Calculate the distances from the input location (lat, lon)
    filtered_df['Distance'] = np.sqrt((filtered_df['StartLat'] - lat) ** 2 + (filtered_df['StartLon'] - lon) ** 2)

    # Find the predicted cluster (the closest one to the input coordinates)
    predicted_cluster_label = filtered_df.loc[filtered_df['Distance'].idxmin(), 'Cluster']

    # Get the nearby clusters (sorted by distance)
    nearby_clusters = filtered_df.groupby('Cluster').apply(lambda x: x['Distance'].min()).sort_values().head(num_nearby_clusters + 1).index

    # Plot the map
    map_center = [lat, lon]
    m = folium.Map(location=map_center, zoom_start=12)

    for cluster_label in nearby_clusters:
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster_label]
        cluster_center_lat = cluster_data['StartLat'].mean()
        cluster_center_lon = cluster_data['StartLon'].mean()
        num_people = len(cluster_data)

        # Mark the nearby clusters
        folium.Marker(
            location=[cluster_center_lat, cluster_center_lon],
            popup=f"Cluster {cluster_label}: {num_people} people",
            icon=folium.Icon(color='blue', icon='info-sign' if cluster_label != predicted_cluster_label else 'star')
        ).add_to(m)

    # Highlight the predicted location with a special marker
    folium.Marker(
        location=[lat, lon],
        popup=f"Predicted location ({lat}, {lon})",
        icon=folium.Icon(color='red', icon='flag')
    ).add_to(m)

    # Save the map
    map_file_name = f'predicted_nearby_clusters_map_{hour}.html'
    m.save(map_file_name)
    print(f"Map saved as {map_file_name}")

# Step 6: Use the prediction and map function

# Example: Predict for coordinates 25.242723, 55.299004 at hour "12"
lat = 25.12497357809829
lon = 55.38183839467802
hour = "12"

# Predict the number of people
predicted_people = predict_num_people(lat, lon, hour)

# Plot the predicted cluster and nearby clusters on the map
plot_predicted_and_nearby_clusters(lat, lon, hour)
