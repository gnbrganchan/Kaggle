import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from dataclasses import dataclass

import matplotlib.pyplot as plt # visualization
plt.rcParams.update({'font.size': 14})
import seaborn as sns # visualization
from tqdm.notebook import tqdm

import warnings # Supress warnings 
warnings.filterwarnings('ignore')

import json
from datetime import datetime
import math

# necessayr to re-check
floor_convert = {'1F' :  0, '2F' : 1, '3F' : 2, '4F' : 3, '5F' : 4, 
                    '6F' : 5, '7F' : 6, '8F' : 7, '9F' : 8,
                    'B'  : -1, 'B1' : -1, 'B2' : -2, 'B3' : -3, 
                    'BF' : -1, 'BM' : -1, 
                    'F1' : 0, 'F2' : 1, 'F3' : 2, 'F4' : 3, 'F5' : 4, 
                    'F6' : 5, 'F7' : 6, 'F8' : 7, 'F9' : 8, 'F10': 9,
                    'L1' : 0, 'L2' : 1, 'L3' : 2, 'L4' : 3, 'L5' : 4, 
                    'L6' : 5, 'L7' : 6, 'L8' : 7, 'L9' : 8, 'L10': 9, 
                    'L11': 10,
                    'G'  : 0, 'LG1': 0, 'LG2': 1, 'LM' : 0, 'M'  : 0, 
                    'P1' : 0, 'P2' : 1,}

@dataclass
class ReadData:
    acce: np.ndarray
    acce_uncali: np.ndarray
    gyro: np.ndarray
    gyro_uncali: np.ndarray
    magn: np.ndarray
    magn_uncali: np.ndarray
    ahrs: np.ndarray
    wifi: np.ndarray
    ibeacon: np.ndarray
    waypoint: np.ndarray


def read_data_file(data_filename):
    acce = []
    acce_uncali = []
    gyro = []
    gyro_uncali = []
    magn = []
    magn_uncali = []
    ahrs = []
    wifi = []
    ibeacon = []
    waypoint = []

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            continue

        line_data = line_data.split('\t')

        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])
            continue
       
        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue
        
        if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
            acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue
        
        if line_data[1] == 'TYPE_GYROSCOPE':
            gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
            gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue
        
        if line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
            magn_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ROTATION_VECTOR':
            ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_WIFI':
            sys_ts = line_data[0]
            ssid = line_data[2]
            bssid = line_data[3]
            rssi = line_data[4]
            lastseen_ts = line_data[6]
            wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
            wifi.append(wifi_data)
            continue

        if line_data[1] == 'TYPE_BEACON':
            ts = line_data[0]
            uuid = line_data[2]
            major = line_data[3]
            minor = line_data[4]
            rssi = line_data[6]
            dist = line_data[7]
            last_seen_timestamp = line_data[9]
            ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi, dist,last_seen_timestamp]
            ibeacon.append(ibeacon_data)
            continue
        
    
    acce = np.array(acce)
    acce_uncali = np.array(acce_uncali)
    gyro = np.array(gyro)
    gyro_uncali = np.array(gyro_uncali)
    magn = np.array(magn)
    magn_uncali = np.array(magn_uncali)
    ahrs = np.array(ahrs)
    wifi = np.array(wifi)
    ibeacon = np.array(ibeacon)
    waypoint = np.array(waypoint)
    
    return ReadData(acce, acce_uncali, gyro, gyro_uncali, magn, magn_uncali, ahrs, wifi, ibeacon, waypoint)
  
  foldername = "../input/indoor-location-navigation/"
df_sample = pd.read_csv(foldername + 'sample_submission.csv')

df_sample["site_id"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[0])
df_sample["path_id"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[1])
df_sample["timestamp"] = df_sample["site_path_timestamp"].apply(lambda x:x.split('_')[2])

list_site = df_sample["site_id"].unique()

if not os.path.exists("train"): os.mkdir("train")
if not os.path.exists("test"): os.mkdir("test")
  
limit_second = 3000

# make train data for each site
for site_id in tqdm(list_site):
    # train mart
    list_train_files = glob.glob(foldername + f"train/{site_id}/**/*.txt", recursive = True)
    df_mart_train = pd.DataFrame()
    df_train_rel = pd.read_parquet(f"../input/mart31/train/{site_id}.parquet").drop(["waypoint_x","waypoint_y","floor"],axis=1)
    df_test_rel = pd.read_parquet(f"../input/mart31/test/{site_id}.parquet")
    for row,filename in enumerate(tqdm(list_train_files)):
        mode,site_id_,floor,path_id = filename.split(".")[2].split("/")[3:]
        floor = floor_convert[floor]
        try: df_all = read_data_file(filename)
        except:continue

        df_waypoint = pd.DataFrame(df_all.waypoint)
        df_waypoint.columns = ['timestamp', 'waypoint_x','waypoint_y']
        df_waypoint["timestamp"] = (df_waypoint["timestamp"]).astype(float)
        df_waypoint = df_waypoint.sort_values('timestamp').reset_index(drop=True)

        df_wifi = pd.DataFrame(df_all.wifi)
        if len(df_wifi) == 0: continue
        df_wifi.columns = ['timestamp', 'ssid', 'bssid', 'rssi', 'last_seen_timestamp']
        df_wifi["timestamp"] = (df_wifi["timestamp"]).astype(float)
        df_wifi["last_seen_timestamp"] = (df_wifi["last_seen_timestamp"]).astype(float)
        df_wifi["rssi"] = (df_wifi["rssi"]).astype(float)
        df_wifi = df_wifi[(df_wifi["timestamp"]-df_wifi["last_seen_timestamp"]) <= limit_second ]
        if len(df_wifi) == 0: continue
        df_wifi = pd.pivot_table(df_wifi, index='timestamp', columns='bssid',values='rssi',aggfunc=np.mean).reset_index()

        df_mart_tmp = df_waypoint.copy()
        df_mart_tmp["path_id"] = path_id
        df_mart_tmp = pd.merge(df_mart_tmp,df_wifi,how="outer",on="timestamp").sort_values('timestamp').reset_index(drop=True)
        df_mart_tmp["path_id"] = path_id
#         sidx = df_mart_tmp[~df_mart_tmp.path_id.isnull()].index[0]
#         eidx = df_mart_tmp[~df_mart_tmp.path_id.isnull()].index[-1]+1
#         df_mart_tmp = df_mart_tmp[sidx:eidx]
#         if row == 1:break
#         df_mart_tmp.index = df_mart_tmp["timestamp"]
#         df_mart_tmp.loc[:,["waypoint_x","waypoint_y"]] = df_mart_tmp.loc[:,["waypoint_x","waypoint_y"]].interpolate('index').reset_index(drop=True).values
#         if row == 3:break
#         df_mart_tmp = df_mart_tmp.loc[df_mart_tmp.index,:]
        df_mart_tmp = df_mart_tmp.merge(df_train_rel,how="left",on=["path_id","timestamp"])
        idx_list = df_mart_tmp[~df_mart_tmp.waypoint_x.isnull()].index.tolist()
        idx_idx = 0
        idx_prev = -1
        idx_next = idx_list[idx_idx]
        for i in range(len(df_mart_tmp)):
            if i == idx_next:
                idx_idx += 1
                idx_prev = i
                if idx_idx < len(idx_list):
                    idx_next = idx_list[idx_idx]
                else: idx_next = -1
            if not math.isnan(df_mart_tmp.at[i,"waypoint_x"]):continue
            if idx_prev == -1:
                df_mart_tmp.at[i,"waypoint_x"] = df_mart_tmp.at[idx_next,"waypoint_x"]+df_mart_tmp.at[i,"rel_x"]-df_mart_tmp.at[idx_next,"rel_x"]   
                df_mart_tmp.at[i,"waypoint_y"] = df_mart_tmp.at[idx_next,"waypoint_y"]+df_mart_tmp.at[i,"rel_y"]-df_mart_tmp.at[idx_next,"rel_y"]        
            elif idx_next == -1:
                df_mart_tmp.at[i,"waypoint_x"] = df_mart_tmp.at[idx_prev,"waypoint_x"]+df_mart_tmp.at[i,"rel_x"]-df_mart_tmp.at[idx_prev,"rel_x"]
                df_mart_tmp.at[i,"waypoint_y"] = df_mart_tmp.at[idx_prev,"waypoint_y"]+df_mart_tmp.at[i,"rel_y"]-df_mart_tmp.at[idx_prev,"rel_y"]
            else:
                df_mart_tmp.at[i,"waypoint_x"] = (df_mart_tmp.at[idx_prev,"waypoint_x"]+df_mart_tmp.at[i,"rel_x"]-df_mart_tmp.at[idx_prev,"rel_x"])/2 +(df_mart_tmp.at[idx_next,"waypoint_x"]+df_mart_tmp.at[i,"rel_x"]-df_mart_tmp.at[idx_next,"rel_x"])/2
                df_mart_tmp.at[i,"waypoint_y"] = (df_mart_tmp.at[idx_prev,"waypoint_y"]+df_mart_tmp.at[i,"rel_y"]-df_mart_tmp.at[idx_prev,"rel_y"])/2 +(df_mart_tmp.at[idx_next,"waypoint_y"]+df_mart_tmp.at[i,"rel_y"]-df_mart_tmp.at[idx_next,"rel_y"])/2
        df_mart_tmp = df_mart_tmp[~df_mart_tmp.index.isin(idx_list)]
        df_mart_tmp["floor"] = floor
        df_mart_train = df_mart_train.append(df_mart_tmp)
#     break
    
    # test mart
    df_sample_site = df_sample.query("site_id == @site_id")
    df_sample_site["timestamp"] = df_sample_site["timestamp"].astype(float)
    list_path = df_sample_site["path_id"].unique()
    df_mart_test = pd.DataFrame()
    for path_id in list_path:
        filename = foldername + f"test/{path_id}.txt"
        df_all = read_data_file(filename)
        df_wifi = pd.DataFrame(df_all.wifi)
        if len(df_wifi) == 0: continue
        df_wifi.columns = ['timestamp', 'ssid', 'bssid', 'rssi', 'last_seen_timestamp']
        df_wifi["timestamp"] = (df_wifi["timestamp"]).astype(float)
        df_wifi["last_seen_timestamp"] = (df_wifi["last_seen_timestamp"]).astype(float)
        df_wifi["rssi"] = (df_wifi["rssi"]).astype(float)
            
        df_ibeacon = pd.DataFrame(df_all.ibeacon)
        if len(df_ibeacon) > 0:
            df_ibeacon.columns = ['timestamp', 'uuid', 'rssi', 'dist','last_seen_timestamp']
            df_ibeacon["timestamp"] = (df_ibeacon["timestamp"]).astype(float)
            df_ibeacon["last_seen_timestamp"] = (df_ibeacon["last_seen_timestamp"]).astype(float)
            time_diff = df_ibeacon.loc[0,"last_seen_timestamp"]-df_ibeacon.loc[0,"timestamp"]
        else:
            time_diff = (df_wifi["last_seen_timestamp"] - df_wifi["timestamp"]).max()
#             time_diff = df_wifi[df_wifi["timestamp"] == df_wifi["timestamp"].min()]["last_seen_timestamp"].max() - df_wifi["timestamp"].min()

        df_wifi["timestamp_raw"] = df_wifi["timestamp"] + time_diff
        df_wifi = df_wifi[(df_wifi["timestamp_raw"]-df_wifi["last_seen_timestamp"]) <= limit_second ]
        df_wifi = pd.pivot_table(df_wifi, index='timestamp', columns='bssid',values='rssi',aggfunc=np.mean).reset_index()

        df_mart_tmp = df_wifi.copy()
        df_mart_tmp["path_id"] = path_id
        df_mart_test = df_mart_test.append(df_mart_tmp)
        
#     break
    # common cols
    common_cols = list(set(df_mart_train.columns) & set(df_mart_test.columns))
    df_mart_train = df_mart_train.loc[:,['waypoint_x', 'waypoint_y', 'floor']+common_cols]
    df_mart_test = df_mart_test.loc[:,common_cols]
    
    df_mart_train.to_parquet(f"train/{site_id}.parquet", index = False) 
    df_mart_test.to_parquet(f"test/{site_id}.parquet", index = False) 
    print(f"site:{site_id} n_cols:{len(common_cols)} n_train:{len(df_mart_train)} n_test:{len(df_mart_test)}")
