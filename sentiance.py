import pandas as pd
import arrow
from io import StringIO
from collections import namedtuple
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree
import numpy as np
from typing import List, Dict
Point = namedtuple('Point', ['latitude', 'longitude'])


def calc_distance(p1: Point, p2: Point) -> float:
    # latitude/longitude are points on a sphere so we need to adjust for curvature while calculating distance,
    # mutliply by 6371000 for converting from angle to meters.
    return (haversine_distances(
        np.array([[p1.latitude, p1.longitude], [p2.latitude, p2.longitude]]) * np.pi / 180.0) * 6371000)[0][1]


def augment_location_df(location_df: pd.DataFrame) -> pd.DataFrame:
    location_df.loc[:, 'point'] = location_df.apply(lambda x: Point(latitude=x['latitude'], longitude=x['longitude']),
                                                    axis=1)
    location_df.loc[:, 'prev_point'] = location_df.loc[:, 'point'].shift(1)
    # need to adjust first point since pandas defaults to Nan which is not a point
    location_df.loc[0, 'prev_point'] = Point(latitude=np.nan, longitude=np.nan)
    location_df.loc[:, 'distance'] = location_df.apply(lambda x: calc_distance(x['point'], x['prev_point']), axis=1)
    return location_df


def read_location_file(file_path: str) -> pd.DataFrame:
    # Reads the person-location csv file and outputs a dataframe
    # I like to keep the return type fixed so even with empty csv we output empty data frame with consistent columns/ order
    # This is usually good for downstream methods.
    return_column_names = ['latitude', 'longitude', 'start_time', 'duration']
    df = pd.read_csv(file_path, sep=';')
    if len(df) == 0:
        return pd.DataFrame(columns=return_column_names)
    time_col_list = [column for column in df.columns if column.startswith('start_time')]
    if len(time_col_list) == 0:
        raise Exception('No start_time column found')
    if len(time_col_list) > 1:
        raise Exception('Multiple start_time columns detected')
    try:
        df.loc[:, 'start_time'] = df.loc[:, time_col_list[0]].map(
            lambda x: pd.Timestamp(arrow.get(x, 'YYYYMMDDHHmmZ').timestamp, unit='s'))
    except arrow.parser.ParseError:
        raise Exception(f'Unknown date/time format in csv. example={df.iloc[0][time_col_list[0]]}')
    df = df.rename(columns={'duration(ms)': 'duration'})
    return df.loc[:, return_column_names]


def test_correct_parse():
    # tests when file format is correct, we get correct output
    test_csv = '''latitude;longitude;start_time(YYYYMMddHHmmZ);duration(ms)
-0.5;0.4;201212121312-0000;1'''
    df = read_location_file(StringIO(test_csv))
    assert (len(df) == 1)
    assert (len(df.columns) == 4)
    assert (df.iloc[0]['longitude'] == 0.4)


def test_timezone_parsing_correct():
    # tests if timezone information is processed. Only difference in two rows in timezone of start_time so
    # drop_duplicates should not drop any row if timezone is correctly processed
    test_csv = '''latitude;longitude;start_time(YYYYMMddHHmmZ);duration(ms)
-0.5;0.4;201212121312-0000;1
-0.5;0.4;201212121312-0200;1'''
    df = read_location_file(StringIO(test_csv))
    assert len(df.drop_duplicates()) == 2


def create_lookup_table(location_df):
    return BallTree(np.array([location_df['latitude'] * np.pi / 180.0, location_df['longitude'] * np.pi / 180.0]).T)


class PersonLocationData(object):
    def __init__(self, sample_df=None, neighbor_tree=None,  tolerance_in_meters=None):
        self.neighbor_tree = neighbor_tree
        self.sample_df = sample_df
        self.tolerance_in_meters = tolerance_in_meters

    def query_nearest_neighbor(self, point:Point, *args,**kwargs):
        return self.neighbor_tree.query(np.array([[point.latitude,point.longitude]])*np.pi/180.0,*args, **kwargs)[1][0][0]

    def query_nearest_neighbor_list(self, point_list:List[Point], *args, **kwargs):
        return self.neighbor_tree.query(np.array(
            [[point.latitude, point.longitude] for point in point_list]) * np.pi / 180.0, *args, **kwargs)[1].reshape(1,-1)
    def query_if_visited(self, point:Point, radius_in_meters, *args, **kwargs):
        return self.neighbor_tree.query_radius(np.array([[point.latitude,point.longitude]])*np.pi/180.0,
                                        r=radius_in_meters/6371000 ,*args, **kwargs)


def person_location_data_builder(file_path:str, tolerance_in_meters:float)->PersonLocationData:
    location_df = read_location_file(file_path)
    location_df = augment_location_df(location_df)
    neighbor_tree = create_lookup_table(location_df)
    return PersonLocationData(sample_df=location_df, neighbor_tree=neighbor_tree,
                              tolerance_in_meters=tolerance_in_meters)

