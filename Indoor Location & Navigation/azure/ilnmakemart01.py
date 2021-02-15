# Blob Storage情報
storage = {
  "account": "ganchanilnsa",
  "container": "ilndata",
  "key": ""
}
 
# マウント先DBFSディレクトリ
mount_point = "/mnt/ilndata/"
 
try:
  # マウント状態のチェック
  mount_dir = mount_point
  if mount_dir[-1] == "/":
    mount_dir = mount_dir[:-1]
  if len(list(filter(lambda x: x.mountPoint == mount_dir, dbutils.fs.mounts()))) > 0:
    print("Already mounted.")
    mounted = True
  else:
    mounted = False
 
  # Blob Storageのマウント
  if not mounted:
    source = "wasbs://{container}@{account}.blob.core.windows.net".format(**storage)
    conf_key = "fs.azure.account.key.{account}.blob.core.windows.net".format(**storage)
 
    mounted = dbutils.fs.mount(
      source=source,
      mount_point = mount_point,
      extra_configs = {conf_key: storage["key"]}
    ) 
 
except Exception as e:
  raise e
 
"mounted: {}".format(mounted)

!pip install jinja2
!pip install tqdm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from dataclasses import dataclass

import matplotlib.pyplot as plt # visualization
plt.rcParams.update({'font.size': 14})
import seaborn as sns # visualization
from tqdm import tqdm

import warnings # Supress warnings 
warnings.filterwarnings('ignore')

import json

foldername = "/dbfs/mnt/ilndata/"
filename = "train/5a0546857ecc773753327266/F1/5e15a2591506f2000638fd57.txt"
mode,site_id,floor,path_id = filename.split(".")[0].split("/")
print(mode,site_id,floor,path_id)

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
            ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
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

sample_file = read_data_file(foldername + "raw/" + filename)

print('acce shape:', sample_file.acce.shape)
print('acce_uncali shape:', sample_file.acce_uncali.shape)
print('gyro shape:', sample_file.gyro.shape)
print('gyro_uncali shape:', sample_file.gyro_uncali.shape)
print('magn shape:', sample_file.magn.shape)
print('magn_uncali shape:',sample_file.magn_uncali.shape)
print('ahrs shape:', sample_file.ahrs.shape)
print('wifi shape:', sample_file.wifi.shape)
print('ibeacon shape:', sample_file.ibeacon.shape)
print('waypoint shape:', sample_file.waypoint.shape)

temp = np.concatenate([sample_file.acce, 
                       sample_file.acce_uncali[:, 1:],
                       sample_file.gyro[:, 1:],
                       sample_file.gyro_uncali[:, 1:],
                       sample_file.magn[:, 1:],
                       sample_file.magn_uncali[:, 1:],
                       sample_file.ahrs[:, 1:],
                      ], axis=1)

imu_df = pd.DataFrame(temp)

imu_df.columns = ['timestamp', 'acce_x','acce_y', 'acce_z','acce_uncali_x','acce_uncali_y', 'acce_uncali_z',
              'gyro_x','gyro_y', 'gyro_z','gyro_uncali_x','gyro_uncali_y', 'gyro_uncali_z',
              'magn_x','magn_y', 'magn_z','magn_uncali_x','magn_uncali_y', 'magn_uncali_z',
              'ahrs_x','ahrs_y', 'ahrs_z']

display(imu_df.head(8).style.set_caption('IMU Data'))

waypoint_df = pd.DataFrame(sample_file.waypoint)
waypoint_df.columns = ['timestamp', 'waypoint_x','waypoint_y']
display(waypoint_df.style.set_caption('Waypoint'))

import glob
list_files = glob.glob("/dbfs/mnt/ilndata/raw/train/**/*.txt", recursive = True)
list_files[0:10]

# clean folder
import shutil
shutil.rmtree('/dbfs/mnt/ilndata/mart01/')
os.mkdir('/dbfs/mnt/ilndata/mart01/')

filename = list_files[414]
mode,site_id,floor,path_id = filename.split(".")[0].split("/")[5:9]
# print(mode,site_id,floor,path_id)
print(filename)
df_wifi = pd.DataFrame(read_data_file(filename).wifi)
# df_wifi.columns = ['timestamp', 'ssid', 'bssid', 'rssi', 'last_seen_timestamp']
# df_wifi["site_id"] = site_id
# df_wifi["floor"] = floor
# df_wifi["path_id"] = path_id
# print(df_wifi.shape)
#   display(df_wifi.head(5))
# list_bssids = df_wifi.bssid.unique()
# print(len(list_bssids))
# for bssid in list_bssids:
# df_out = df_wifi.query("bssid == @bssid")
# outfilename = f"/dbfs/mnt/ilndata/mart01/{bssid}.csv"
# if os.path.exists(outfilename):
#   df = pd.read_csv(outfilename)
#   df_out = df.append(df_out)
# #     df_out.to_csv(outfilename, index = False)
# #       df_out.to_csv(outfilename, index = False, mode="a", header=False)
# else:
# #     df_out.to_csv(outfilename, index = False)
# display(df_wifi)

# filename = list_files[0]
for filename in tqdm(list_files):
  mode,site_id,floor,path_id = filename.split(".")[0].split("/")[5:9]
  # print(mode,site_id,floor,path_id)
  df_wifi = pd.DataFrame(read_data_file(filename).wifi)
  if len(df_wifi) == 0: continue
  df_wifi.columns = ['timestamp', 'ssid', 'bssid', 'rssi', 'last_seen_timestamp']
  df_wifi["site_id"] = site_id
  df_wifi["floor"] = floor
  df_wifi["path_id"] = path_id
  # print(df_wifi.shape)
#   display(df_wifi.head(5))
  list_bssids = df_wifi.bssid.unique()
  # print(len(list_bssids))
  for bssid in list_bssids:
    df_out = df_wifi.query("bssid == @bssid")
    outfilename = f"/dbfs/mnt/ilndata/mart01/{bssid}.csv"
    if os.path.exists(outfilename):
      df = pd.read_csv(outfilename)
      df_out = df.append(df_out)
      df_out.to_csv(outfilename, index = False)
#       df_out.to_csv(outfilename, index = False, mode="a", header=False)
    else:
      df_out.to_csv(outfilename, index = False)
      
dbutils.fs.unmount("/mnt/ilndata")
