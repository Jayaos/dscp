import os
import numpy as np
import pandas as pd
from pathlib import Path


class BasePredictorData:
    """
    Data class to load data for base predictor
    """

    def __init__(self):
        self.data_type = None
        self.data = None

    def load_data(self, data_type, data_dir):

        self.data_type = data_type

        if data_type in ["nsdb-60m", "nsdb-30m"]:
            self._load_nsdb_data_from_data_dir(data_type, data_dir)

        elif data_type in ["air-25", "air-10"]:
            self._load_beijing_air_data_from_data_dir(data_type, data_dir)

        elif data_type in ["toy"]:
            self._load_toy_data(data_dir)

    def _load_toy_data(self, data_dir):

        data_path = Path(data_dir)
        if data_path.is_dir():
            csv_paths = sorted(data_path.glob("*.csv"))
            if not csv_paths:
                raise FileNotFoundError(f"No toy CSV files found in {data_path}.")
        else:
            csv_paths = [data_path]

        data = {}
        for csv_path in csv_paths:
            raw_data = pd.read_csv(csv_path)
            if "Y" not in raw_data:
                raise ValueError(f"Toy CSV must contain a 'Y' column: {csv_path}")

            if "X" in raw_data:
                x = raw_data.loc[:, raw_data.columns.str.startswith("X")]
            else:
                # HopCPT toy files use an unnamed index column, Y, then feature columns.
                x = raw_data.iloc[:, 2:]

            if x.shape[1] == 0:
                raise ValueError(f"Toy CSV must contain at least one feature column: {csv_path}")

            data[csv_path.stem] = {
                "x": x.to_numpy(dtype=np.float32),
                "y": raw_data["Y"].to_numpy(dtype=np.float32),
            }
        
        self.data = data

    def _load_nsdb_data_from_data_dir(self, data_type, data_dir):

        listdir = os.listdir(data_dir)
        data = dict()

        for file_name in listdir:
            if file_name.endswith("csv"):
                x, y = self.load_nsdb_data(data_type, os.path.join(data_dir, file_name))
                data[file_name.removesuffix(".csv")] = {"x" : x,
                                                        "y" : y}
        
        self.data = data

    def _load_beijing_air_data_from_data_dir(self, data_type, data_dir):

        listdir = os.listdir(data_dir)
        data = dict()

        for file_name in listdir:
            if file_name.endswith("csv"):
                x, y = self.load_bejing_air_data(data_type, os.path.join(data_dir, file_name))
                data[file_name.removesuffix(".csv")] = {"x" : x,
                                                        "y" : y}
        
        self.data = data 

    @staticmethod
    def load_nsdb_data(dataset_type, data_path):
        
        data = pd.read_csv(data_path)
        if dataset_type.startswith("nsdb-60m"):
            drop_every_n = 2
        elif dataset_type.startswith("nsdb-30m"):
            drop_every_n = 1
        else:
            raise ValueError("Type not supported")
            
        append_coords = dataset_type.endswith("-wCoord")
        data = data.iloc[::drop_every_n, :]
        Y_full = data['dhi']
        X_full = data.loc[:, data.columns != 'dhi']
        X_full.drop(columns=X_full.columns[0:6], inplace=True)  # Drop Date Stuff
        if append_coords:
            p = Path(data_path)
            coordinates = pd.read_csv(p.parent.parent / "solar_coordinates.csv")
            lat = coordinates[coordinates["location"] == p.stem]['latitude'].item()
            long = coordinates[coordinates["location"] == p.stem]['longitude'].item()
            X_full["latitude"] = lat
            X_full["longitude"] = long

        return X_full.to_numpy(), Y_full.to_numpy()

    @staticmethod
    def load_bejing_air_data(dataset_type, data_path, wd_encode='encode', imputer=None):
        data = pd.read_csv(data_path)
        if 'Unnamed: 0' in data:
            data.drop(columns=['Unnamed: 0'], inplace=True, axis=1)
        data.drop(columns=['No', 'station'], inplace=True, axis=1)
        data.drop(columns=['year', 'month', 'day', 'hour'], inplace=True, axis=1)
        data_wd = data['wd'].fillna(value="Unknown")
        if imputer is not None:
            data = pd.DataFrame(imputer.fit_transform(data.loc[:, data.columns != 'wd']))
        else:
            data = data.loc[:, data.columns != 'wd'].ffill()
            data = data.bfill()
        data['wd'] = data_wd
        if dataset_type.startswith("air-25"):
            Y_full = data['PM2.5']
        elif dataset_type.startswith("air-10"):
            Y_full = data['PM10']
        data.drop(columns=['PM2.5', 'PM10'], inplace=True, axis=1)
        if wd_encode == "drop":
            data.drop(columns=["wd"], inplace=True, axis=1)
        elif wd_encode == "one-hot":
            data = pd.get_dummies(data)
        elif wd_encode == 'encode':
            data['wd_h'] = data['wd'].apply(lambda x: _encode_direction(x, True))
            data['wd_v'] = data['wd'].apply(lambda x: _encode_direction(x, False))
            data.drop(columns=["wd"], inplace=True, axis=1)
        X_full = data

        return X_full.to_numpy(), Y_full.to_numpy()
    
    
def _encode_direction(direction, horizontal):
    if horizontal:
        if direction in ['N', 'S']:
            return 0
        elif direction in ['NNW', 'SSW']:
            return -0.5
        elif direction in ['NW', 'SW']:
            return -0.7
        elif direction in ['WNW', 'WSW']:
            return -0.86
        elif direction == 'W':
            return -1
        elif direction in ['NNE', 'SSE']:
            return 0.5
        elif direction in ['NE', 'SE']:
            return 0.7
        elif direction in ['ENE', 'ESE']:
            return 0.86
        elif direction == 'E':
            return 1
        elif direction == 'Unknown':
            return 0
        else:
            raise ValueError("Invalid Dir")
    else:
        if direction in ['W', 'E']:
            return 0
        elif direction in ['WSW', 'ESE']:
            return -0.5
        elif direction in ['SW', 'SE']:
            return -0.7
        elif direction in ['SSW', 'SSE']:
            return -0.86
        elif direction == 'S':
            return -1
        elif direction in ['WNW', 'ENE']:
            return 0.5
        elif direction in ['NW', 'NE']:
            return 0.7
        elif direction in ['NNW', 'NNE']:
            return 0.86
        elif direction == 'N':
            return 1
        elif direction == 'Unknown':
            return 0
        else:
            raise ValueError("Invalid Dir")

