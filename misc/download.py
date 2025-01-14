import sys, os, subprocess
from subprocess import check_output


depth_data_list = [
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip",
]

raw_data_list = [
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0011/2011_09_26_drive_0011_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0013/2011_09_26_drive_0013_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0014/2011_09_26_drive_0014_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0015/2011_09_26_drive_0015_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0017/2011_09_26_drive_0017_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0018/2011_09_26_drive_0018_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0019/2011_09_26_drive_0019_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0022/2011_09_26_drive_0022_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0023/2011_09_26_drive_0023_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0027/2011_09_26_drive_0027_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0028/2011_09_26_drive_0028_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0029/2011_09_26_drive_0029_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0032/2011_09_26_drive_0032_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0035/2011_09_26_drive_0035_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0036/2011_09_26_drive_0036_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0039/2011_09_26_drive_0039_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0046/2011_09_26_drive_0046_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0048/2011_09_26_drive_0048_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0051/2011_09_26_drive_0051_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0052/2011_09_26_drive_0052_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0056/2011_09_26_drive_0056_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0057/2011_09_26_drive_0057_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0059/2011_09_26_drive_0059_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0060/2011_09_26_drive_0060_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0061/2011_09_26_drive_0061_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0064/2011_09_26_drive_0064_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0070/2011_09_26_drive_0070_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0079/2011_09_26_drive_0079_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0084/2011_09_26_drive_0084_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0086/2011_09_26_drive_0086_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0087/2011_09_26_drive_0087_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0091/2011_09_26_drive_0091_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0093/2011_09_26_drive_0093_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0096/2011_09_26_drive_0096_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0101/2011_09_26_drive_0101_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0104/2011_09_26_drive_0104_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0106/2011_09_26_drive_0106_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0113/2011_09_26_drive_0113_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0117/2011_09_26_drive_0117_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0119/2011_09_26_drive_0119_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0001/2011_09_28_drive_0001_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0002/2011_09_28_drive_0002_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0016/2011_09_28_drive_0016_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0021/2011_09_28_drive_0021_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0034/2011_09_28_drive_0034_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0035/2011_09_28_drive_0035_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0037/2011_09_28_drive_0037_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0038/2011_09_28_drive_0038_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0039/2011_09_28_drive_0039_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0043/2011_09_28_drive_0043_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0045/2011_09_28_drive_0045_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0047/2011_09_28_drive_0047_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0053/2011_09_28_drive_0053_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0054/2011_09_28_drive_0054_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0057/2011_09_28_drive_0057_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0065/2011_09_28_drive_0065_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0066/2011_09_28_drive_0066_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0068/2011_09_28_drive_0068_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0070/2011_09_28_drive_0070_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0071/2011_09_28_drive_0071_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0075/2011_09_28_drive_0075_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0077/2011_09_28_drive_0077_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0078/2011_09_28_drive_0078_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0080/2011_09_28_drive_0080_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0082/2011_09_28_drive_0082_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0086/2011_09_28_drive_0086_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0087/2011_09_28_drive_0087_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0089/2011_09_28_drive_0089_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0090/2011_09_28_drive_0090_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0094/2011_09_28_drive_0094_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0095/2011_09_28_drive_0095_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0096/2011_09_28_drive_0096_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0098/2011_09_28_drive_0098_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0100/2011_09_28_drive_0100_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0102/2011_09_28_drive_0102_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0103/2011_09_28_drive_0103_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0104/2011_09_28_drive_0104_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0106/2011_09_28_drive_0106_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0108/2011_09_28_drive_0108_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0110/2011_09_28_drive_0110_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0113/2011_09_28_drive_0113_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0117/2011_09_28_drive_0117_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0119/2011_09_28_drive_0119_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0121/2011_09_28_drive_0121_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0122/2011_09_28_drive_0122_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0125/2011_09_28_drive_0125_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0126/2011_09_28_drive_0126_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0128/2011_09_28_drive_0128_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0132/2011_09_28_drive_0132_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0134/2011_09_28_drive_0134_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0135/2011_09_28_drive_0135_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0136/2011_09_28_drive_0136_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0138/2011_09_28_drive_0138_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0141/2011_09_28_drive_0141_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0143/2011_09_28_drive_0143_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0145/2011_09_28_drive_0145_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0146/2011_09_28_drive_0146_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0149/2011_09_28_drive_0149_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0153/2011_09_28_drive_0153_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0154/2011_09_28_drive_0154_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0155/2011_09_28_drive_0155_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0156/2011_09_28_drive_0156_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0160/2011_09_28_drive_0160_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0161/2011_09_28_drive_0161_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0162/2011_09_28_drive_0162_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0165/2011_09_28_drive_0165_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0166/2011_09_28_drive_0166_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0167/2011_09_28_drive_0167_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0168/2011_09_28_drive_0168_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0171/2011_09_28_drive_0171_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0174/2011_09_28_drive_0174_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0177/2011_09_28_drive_0177_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0179/2011_09_28_drive_0179_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0183/2011_09_28_drive_0183_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0184/2011_09_28_drive_0184_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0185/2011_09_28_drive_0185_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0186/2011_09_28_drive_0186_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0187/2011_09_28_drive_0187_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0191/2011_09_28_drive_0191_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0192/2011_09_28_drive_0192_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0195/2011_09_28_drive_0195_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0198/2011_09_28_drive_0198_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0199/2011_09_28_drive_0199_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0201/2011_09_28_drive_0201_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0204/2011_09_28_drive_0204_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0205/2011_09_28_drive_0205_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0208/2011_09_28_drive_0208_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0209/2011_09_28_drive_0209_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0214/2011_09_28_drive_0214_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0216/2011_09_28_drive_0216_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0220/2011_09_28_drive_0220_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0222/2011_09_28_drive_0222_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_drive_0225/2011_09_28_drive_0225_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0004/2011_09_29_drive_0004_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0026/2011_09_29_drive_0026_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0071/2011_09_29_drive_0071_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_drive_0108/2011_09_29_drive_0108_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0018/2011_09_30_drive_0018_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0020/2011_09_30_drive_0020_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0028/2011_09_30_drive_0028_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0033/2011_09_30_drive_0033_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0034/2011_09_30_drive_0034_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0072/2011_09_30_drive_0072_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0034/2011_10_03_drive_0034_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0042/2011_10_03_drive_0042_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0058/2011_10_03_drive_0058_sync.zip",
]

data_scene_flow = [
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_multiview.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_scene_flow.zip",
]


def download_KITTI():
    try:
        check_output(["man", "mkdir"])
        check_output(["man", "wget"])
        check_output(["man", "unzip"])
        check_output(["man", "rsync"])
        check_output(["man", "rm"])
        check_output(["man", "mv"])
        check_output(["man", "echo"])
    except Exception as e:
        print("Program not installed", e)
        print(
            "Make sure to install all of these Programs: man, mkdir, wget, unzip, rsync, rm, mv, echo."
        )
        sys.exit(os.EX_OK)

    subprocess.run("mkdir -p ./data/KITTI", shell=True)

    for item in depth_data_list:
        namewithzip = item.split("/")[-1]
        name = namewithzip[:-4]
        subprocess.run("mkdir -p temp", shell=True)
        subprocess.run(f"wget {item} -nc -O ./temp/{namewithzip}", shell=True)
        subprocess.run(
            f"unzip -o ./temp/{namewithzip} -d ./temp/{name}",
            shell=True,
        )
        rsync_cmd = f"rsync -avhP ./temp/{name}/train/ ./data/KITTI"
        subprocess.run(rsync_cmd, shell=True)
        rsync_cmd = f"rsync -avhP ./temp/{name}/val/ ./data/KITTI"
        subprocess.run(rsync_cmd, shell=True)
        subprocess.run(
            "rm -rf ./temp && echo rm worked || echo rm did not work", shell=True
        )

    for item in raw_data_list:
        namewithzip = item.split("/")[-1]
        name = namewithzip[:-4]

        subprocess.run("mkdir -p temp", shell=True)
        subprocess.run(f"wget {item} -O ./temp/{namewithzip}", shell=True)
        subprocess.run(f"unzip -o ./temp/{namewithzip} -d ./temp/{name}", shell=True)

        date = next(
            d
            for d in (next(os.walk(os.path.join("./temp", name)))[1])
            if not d[0] == "."
        )

        if "calib" in item:
            calib_dir = os.path.join("./temp", name, date)
            for file in os.listdir(calib_dir):
                if os.path.isfile(os.path.join(calib_dir, file)):
                    mv_cmd = f"mv {calib_dir}/{file} ./data/KITTI/{date}_{file}"
                    subprocess.run(mv_cmd, shell=True)

        else:
            rsync_cmd = f"rsync -avhP ./temp/{name}/{date}/{name} ./data/KITTI"
            subprocess.run(rsync_cmd, shell=True)

        subprocess.run(
            "rm -rf ./temp && echo rm worked || echo rm did not work", shell=True
        )

    return 0


def download_KITTI2015():
    try:
        check_output(["man", "mkdir"])
        check_output(["man", "wget"])
        check_output(["man", "unzip"])
        check_output(["man", "rsync"])
        check_output(["man", "rm"])
        check_output(["man", "mv"])
        check_output(["man", "echo"])
    except Exception as e:
        print("Program not installed", e)
        print(
            "Make sure to install all of these Programs: man, mkdir, wget, unzip, rsync, rm, mv, echo."
        )
        sys.exit(os.EX_OK)

    subprocess.run("mkdir -p ./data/KITTI2015", shell=True)

    for item in data_scene_flow:
        namewithzip = item.split("/")[-1]
        name = namewithzip[:-4]
        subprocess.run("mkdir -p temp", shell=True)
        subprocess.run(f"wget {item} -nc -O ./temp/{namewithzip}", shell=True)
        subprocess.run(
            f"unzip -o ./temp/{namewithzip} -d ./temp/{name}",
            shell=True,
        )
        rsync_cmd = f"rsync -avhP ./temp/{name}/ ./data/KITTI2015"
        subprocess.run(rsync_cmd, shell=True)
        subprocess.run(
            "rm -rf ./temp && echo rm worked || echo rm did not work", shell=True
        )


def check_kitti_availability(args):
    dataset_path = os.path.join(args.data_directory, args.dataset)
    if not os.path.exists(dataset_path) and args.dataset == "KITTI":
        print(f"No data found at {dataset_path}.")
        if (
            input(f"Would you like to download {args.dataset} dataset? (y/n): ")
            .lower()
            .strip()[:1]
            == "y"
        ):
            download_KITTI()

    if args.modus_operandi == "train":
        validation_dataset_path = os.path.join(
            args.data_directory, args.validation_dataset
        )
        if (
            not os.path.exists(validation_dataset_path)
            and args.validation_dataset == "KITTI2015"
        ):
            print(f"No data found at {validation_dataset_path}.")
            if (
                input(
                    f"Would you like to download {args.validation_dataset} dataset? (y/n): "
                )
                .lower()
                .strip()[:1]
                == "y"
            ):
                download_KITTI2015()

    if not os.path.exists(dataset_path):
        print(f"Program aborts, as no data could be found at {dataset_path}.")
        sys.exit(1)
