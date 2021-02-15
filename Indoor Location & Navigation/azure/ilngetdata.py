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

!pip install kaggle

import json
import os
if not os.path.exists("/home/root/.kaggle"):
  os.mkdir("/home/root/.kaggle")
with open("/home/root/.kaggle/kaggle.json","w") as f:
  json.dump({"username":"iwatatakuya","key":"28d1268f8de343cf7adfddd92aca43f9"}, f)
  
if not os.path.exists("/dbfs/mnt/ilndata/zip/"):
  os.mkdir("/dbfs/mnt/ilndata/zip")
if not os.path.exists("/dbfs/mnt/ilndata/raw"/):
  os.mkdir("/dbfs/mnt/ilndata/raw")
  
!kaggle competitions download -c indoor-location-navigation -p /dbfs/mnt/ilndata/zip/

!unzip /dbfs/mnt/ilndata/zip/indoor-location-navigation.zip -d /dbfs/mnt/ilndata/raw/

dbutils.fs.unmount("/mnt/ilndata")
