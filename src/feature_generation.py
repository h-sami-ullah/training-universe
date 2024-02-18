import numpy as np
import pandas as pd
import collections.abc
from math import radians
import h3
from sklearn.preprocessing import LabelEncoder


class FeatureExtractor:

    def __init__(self, df: pd.DataFrame, restaurants_ids: dict):
        """
        Initializes the FeatureExtractor with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to generate features from.
        """
        self.df = df
        self.restaurants_ids = restaurants_ids

    def calc_dist(self, p1x, p1y, p2x, p2y):
        """
        Calculates Euclidean distances between two points.

        Args:
            p1x, p1y: Coordinates of the first point.
            p2x, p2y: Coordinates of the second point.

        Returns:
            Distance between the two points.
        """
        p1 = (p2x - p1x) ** 2
        p2 = (p2y - p1y) ** 2
        dist = np.sqrt(p1 + p2)
        return dist.tolist() if isinstance(p1x, collections.abc.Sequence) else dist

    def add_dist_to_restaurant_feature(self):
        """Calculates and adds the Euclidean distance to the restaurant for each order."""
        self.df['dist_to_restaurant'] = self.calc_dist(
            self.df.courier_lat, self.df.courier_lon,
            self.df.restaurant_lat, self.df.restaurant_lon
        )

    def avg_dist_to_restaurants(self, courier_lat, courier_lon):
        """
        Calculates the average distance from a courier location to all restaurants.

        Parameters:
        - courier_lat (float): Latitude of the courier location.
        - courier_lon (float): Longitude of the courier location.

        Returns:
        float: The average distance to restaurants from the courier location.
        """
        distances = [self.calc_dist(v['lat'], v['lon'], courier_lat, courier_lon)
                     for v in self.restaurants_ids.values()]
        return np.mean(distances)

    def add_avg_dist_to_restaurants_feature(self):
        """
        Calculates and adds the average distance to restaurants feature for each courier.
        """
        self.df['avg_dist_to_restaurants'] = [
            self.avg_dist_to_restaurants(lat, lon)
            for lat, lon in zip(self.df['courier_lat'], self.df['courier_lon'])
        ]

    def calc_haversine_dist(self, lat1, lon1, lat2, lon2):
        """
        Calculates the Haversine distance between two points on the Earth.
        """
        R = 6372.8  # Earth radius in kilometers
        if isinstance(lat1, collections.abc.Sequence):
            dLat = np.array([radians(l2 - l1) for l2, l1 in zip(lat2, lat1)])
            dLon = np.array([radians(l2 - l1) for l2, l1 in zip(lon2, lon1)])
            lat1 = np.array([radians(l) for l in lat1])
            lat2 = np.array([radians(l) for l in lat2])
        else:
            dLat = radians(lat2 - lat1)
            dLon = radians(lon2 - lon1)
            lat1 = radians(lat1)
            lat2 = radians(lat2)

        a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        dist = R * c
        return dist.tolist() if isinstance(lon1, collections.abc.Sequence) else dist

    def add_haversine_dist_to_restaurant(self):
        """Calculates and adds the Euclidean distance to the restaurant for each order."""
        self.df['Hdist_to_restaurant'] = self.calc_haversine_dist(self.df.courier_lat.tolist(),
                                                                  self.df.courier_lon.tolist(),
                                                                  self.df.restaurant_lat.tolist(),
                                                                  self.df.restaurant_lon.tolist())

    def avg_Hdist_to_restaurants(self, courier_lat, courier_lon):
        """
        Calculates the average Haversine distance from a courier location to all restaurants.
        """
        distances = [self.calc_haversine_dist(courier_lat, courier_lon, v['lat'], v['lon'])
                     for v in self.restaurants_ids.values()]
        return np.mean(distances)

    def add_avg_haversine_distance_feature(self):
        """
        Adds average Haversine distance to restaurants feature for each courier.
        """
        self.df['avg_Hdist_to_restaurants'] = [
            self.avg_Hdist_to_restaurants(lat, lon) for lat, lon in
            zip(self.df.courier_lat, self.df.courier_lon)
        ]

    def initiate_centroids(self, k):
        """
        Selects k data points as centroids.

        Parameters:
        - k (int): Number of centroids to select.

        Returns:
        pd.DataFrame: Dataframe containing the selected centroids.
        """
        df_restaurants = pd.DataFrame([{"lat": v['lat'], "lon": v['lon']} for v in self.restaurants_ids.values()])
        centroids = df_restaurants.sample(k)
        return centroids

    def centroid_assignation(self, centroids):
        """
        Assigns each data point to the nearest centroid.

        Parameters:
        - centroids (pd.DataFrame): Dataframe containing the centroids.

        Returns:
        pd.DataFrame: Updated dataframe with centroid assignment.
        """

        k = len(centroids)
        n = len(self.df)
        assignation = []
        assign_errors = []
        centroids_list = [c for i, c in centroids.iterrows()]
        for i, obs in self.df.iterrows():
            # Estimate error
            all_errors = [self.calc_dist(centroid['lat'], centroid['lon'], obs['courier_lat'], obs['courier_lon'])
                          for centroid in centroids_list]

            # Get the nearest centroid and the error
            nearest_centroid = np.where(all_errors == np.min(all_errors))[0].tolist()[0]
            nearest_centroid_error = np.min(all_errors)

            # Add values to corresponding lists
            assignation.append(nearest_centroid)
            assign_errors.append(nearest_centroid_error)
        return assignation, assign_errors

    def add_clusters_embedding_feature(self, k: int = 5):
        """
        Assigns each data point to the nearest centroid and adds the corresponding cluster embedding feature.
        """
        np.random.seed(1)
        centroids = self.initiate_centroids(k=k)
        assignation, assign_errors = self.centroid_assignation(centroids)
        self.df['Five_Clusters_embedding'] = assignation
        self.df['Five_Clusters_embedding_error'] = assign_errors

    def add_timestamp_to_datetime(self, column_name: str) -> None:
        """
        Parses datetime values in the specified column, trying two formats: '%Y-%m-%dT%H:%M:%S.%fZ'
        and '%Y-%m-%dT%H:%M:%SZ'. If the first format fails for certain values, it tries the second
        format.

        Parameters:
        - column_name (str): Name of the column containing datetime values to be parsed.

        Returns:
        None
        """
        # Parse datetime column with the first format
        data1 = pd.to_datetime(self.df[column_name], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')

        # Get indices of NaT values
        na_values_indices = data1[data1.isna()].index

        # Extract values with NaT
        na_values = self.df.loc[na_values_indices, column_name]

        # Parse NaT values with the second format
        data2 = pd.to_datetime(na_values, format='%Y-%m-%dT%H:%M:%SZ')

        # Update DataFrame with parsed values
        self.df.loc[na_values_indices, column_name] = data2

        # Update DataFrame with parsed values from the first format
        non_na_indices = data1[data1.notna()].index
        self.df.loc[non_na_indices, column_name] = data1[non_na_indices]
        self.df[column_name] = pd.to_datetime(self.df[column_name])

    def add_h3_index_feature(self, resolution: int = 7):
        """
        Adds H3 index feature based on courier latitude and longitude.

        Parameters:
        - resolution (int): Resolution level for H3 index (default is 7).
        """
        self.df['h3_index'] = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in
                               zip(self.df['courier_lat'], self.df['courier_lon'])]

    def add_date_day_hour_features(self):
        """
        Adds features for day number and hour number based on courier location timestamp.
        """
        self.add_timestamp_to_datetime('courier_location_timestamp')
        self.add_timestamp_to_datetime('order_created_timestamp')
        self.df['date_day_number'] = self.df['courier_location_timestamp'].dt.dayofyear
        self.df['date_hour_number'] = self.df['courier_location_timestamp'].dt.hour

    def add_orders_busyness_by_h3_hour_feature(self):
        """
        Adds a feature indicating the orders busyness by H3 index and hour of the day.
        """
        index_list = [(i, d, hr) for (i, d, hr) in
                      zip(self.df['h3_index'], self.df['date_day_number'], self.df['date_hour_number'])]

        set_indexes = list(set(index_list))
        dict_indexes = {label: index_list.count(label) for label in set_indexes}

        self.df['orders_busyness_by_h3_hour'] = [dict_indexes[i] for i in index_list]

    def add_restaurants_per_index_feature(self):
        """
        Adds a feature indicating the number of restaurants per H3 index.
        """
        restaurants_counts_per_h3_index = {
            a: len(b) for a, b in zip(
                self.df.groupby('h3_index')['restaurant_id'].unique().index,
                self.df.groupby('h3_index')['restaurant_id'].unique()
            )
        }
        self.df['restaurants_per_index'] = [restaurants_counts_per_h3_index[h] for h in self.df['h3_index']]

    def encode_categorical_features(self):
        """
        Encodes categorical features in the DataFrame using LabelEncoder.
        """
        columns_to_encode = list(self.df.select_dtypes(include=['category', 'object']))
        le = LabelEncoder()
        for feature in columns_to_encode:
            try:
                self.df[feature] = le.fit_transform(self.df[feature])
            except:
                print('Error encoding ' + feature)

    def h3_index_feature_set_type(self):
        """
        Converts the 'h3_index' column to categorical type.
        """
        self.df['h3_index'] = self.df['h3_index'].astype('category')

    def save_dataframe(self, file_path: str):
        """
        Saves the DataFrame to a CSV file.

        Parameters:
        - file_path (str): The file path to save the CSV file.
        """
        self.df.to_csv(file_path, index=False)

    def generate_features(self):
        """
        Public method to apply all feature generation methods.

        """
        self.add_dist_to_restaurant_feature()
        self.add_avg_dist_to_restaurants_feature()
        self.add_haversine_dist_to_restaurant()
        self.add_avg_haversine_distance_feature()
        self.add_clusters_embedding_feature()
        self.add_h3_index_feature()
        self.add_date_day_hour_features()
        self.add_orders_busyness_by_h3_hour_feature()
        self.add_restaurants_per_index_feature()
        self.h3_index_feature_set_type()
        self.encode_categorical_features()

        # Call other feature generation methods here


if __name__ == "__main__":
    # Example usage
    from data_collection import *

    data_collector = DataCollection(r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook\final_dataset (2).csv")
    dataframe = data_collector.get_dataframe()
    restaurants_ids = data_collector.resturants_ids
    print(dataframe.head())
    fe = FeatureExtractor(dataframe, restaurants_ids)
    fe.generate_features()
    print(fe.df.head())
    print(restaurants_ids)
    output_file_path = r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook\output_dataframe.csv"
    fe.save_dataframe(output_file_path)
    print(f"DataFrame saved to {output_file_path}")
