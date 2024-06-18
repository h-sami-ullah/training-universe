import pandas as pd
import logging


class DataCollection:
    def __init__(
        self, file_path: str, dropna_axis: int = 0, dropna_inplace: bool = True
    ) -> None:
        """
        Initializes the DataCollection object by loading and optionally cleaning data from a CSV file, and generating
        unique restaurant IDs.

        Args:
            file_path (str): The file path to the CSV data file.
            dropna_axis (int, optional): Axis along which to drop missing values.
                                         0 drops rows which contain missing values.
                                         1 drops columns which contain missing value.
                                         Defaults to 0.
            dropna_inplace (bool, optional): Whether to modify the DataFrame in place
                                              or return a copy with dropped rows/columns.
                                              Defaults to True.
        """
        self.df = None
        try:
            self.df = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully from {file_path}.")
        except FileNotFoundError as e:
            logging.error(
                f"File not found at {file_path}. Please check the file path and try again. Error: {e}"
            )
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading data: {e}")
            raise

        if self.df is not None:
            self.df.dropna(axis=dropna_axis, inplace=dropna_inplace)
            logging.info("Missing values dropped from DataFrame.")
        self.restaurants_ids = None
        self.generate_restaurant_ids()  # Automatically generate restaurant IDs upon initialization

    def generate_restaurant_ids(self) -> None:
        """
        Generates unique restaurant IDs based on latitude and longitude and labels
        each row in the dataframe with the corresponding restaurant ID.
        """
        if self.df is None:
            logging.error("DataFrame is not initialized.")
            return

        restaurants_ids = {}
        for lat, lon in zip(self.df["restaurant_lat"], self.df["restaurant_lon"]):
            id = f"{lat}_{lon}"
            if id not in restaurants_ids:
                restaurants_ids[id] = {"lat": lat, "lon": lon}

        for i, key in enumerate(restaurants_ids.keys()):
            restaurants_ids[key]["id"] = i

        self.df["restaurant_id"] = [
            restaurants_ids[f"{lat}_{lon}"]["id"]
            for lat, lon in zip(self.df["restaurant_lat"], self.df["restaurant_lon"])
        ]
        self.restaurants_ids = restaurants_ids
        logging.info("Unique restaurant IDs generated.")

    def get_unique_couriers(self) -> int:
        """
        Returns the number of unique courier IDs in the dataset.

        Returns:
            int: The count of unique courier IDs.
        """

        if self.df is None:
            logging.error(
                "DataFrame is not initialized. Unable to determine unique couriers."
            )
            return 0
        return len(self.df.courier_id.unique())

    def get_unique_restaurants(self) -> int:
        """
        Returns the number of unique restaurants, assuming that `generate_restaurant_ids`
        has already been called during initialization.

        Returns:
            int: The count of unique restaurants.
        """
        if self.df is None:
            logging.error(
                "DataFrame is not initialized. Unable to determine unique restaurants."
            )
            return 0
        return len(self.df["restaurant_id"].unique())

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the processed DataFrame.

        Returns:
            pd.DataFrame: The processed DataFrame with restaurant IDs.
        """
        if self.df is None:
            logging.error("DataFrame is not initialized.")
            return pd.DataFrame()  # Return an empty DataFrame as a safe fallback
        return self.df

    def save_dataframe(self, file_path: str):
        """
        Saves the DataFrame to a CSV file.

        Args:
            file_path (str): The file path to save the CSV file.
        """
        if self.df is None:
            logging.error("DataFrame is not initialized. Unable to save.")
            return
        self.df.to_csv(file_path, index=False)
        logging.info(f"DataFrame saved to {file_path}.")


if __name__ == "__main__":
    # Example usage
    data_collector = DataCollection(
        r"C:\Users\Hafiz\Downloads\Laptop\Personal\Notebook\final_dataset (2).csv"
    )
    dataframe = data_collector.get_dataframe()
    print(dataframe.head())
