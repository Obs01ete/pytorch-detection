import os
import zipfile
import urllib.request
import multiprocessing


files = [
    "data_object_calib.zip",
    "data_object_label_2.zip",
    "data_object_image_2.zip",
    # "data_object_velodyne.zip",
]

# files = [
#     "data_tracking_velodyne.zip",
#     "data_tracking_image_2.zip",
#     "data_tracking_oxts.zip",
#     "data_tracking_calib.zip",
#     "data_tracking_label_2.zip",
# ]

# files = [
#     "data_odometry_velodyne.zip",
# ]

location = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"

dst_dir = "kitti/"
os.makedirs(dst_dir, exist_ok=True)


def download(file):
    url = location + file
    dst_file = os.path.join(dst_dir, file)

    if not os.path.exists(dst_file):
        print("Downloading", url)
        urllib.request.urlretrieve(url, dst_file)

    print("Unzipping", url)

    with zipfile.ZipFile(dst_file, 'r') as zip_ref:
        zip_ref.extractall(dst_dir)

    print("Done", url)


def main():
    pool = multiprocessing.Pool(len(files))
    pool.map(download, files)

    print("Done!")


if __name__ == "__main__":
    main()
