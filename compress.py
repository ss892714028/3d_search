import os
from collections import deque
import pymeshlab as ml
from config import SEARCH_FEATURE_PATH, LOAD_FEATURE_PATH, UPLOAD_PATH, DATA_PATH
import getopt
import sys
from logs import LOGGER


def process(data_root, out_root, file_name):
    """
    Preprocess and compress a single 3d models
    Args:
        data_root ([str]): path of where the 3d models are originally stored
        out_root ([str]): path of where the compressed 3d models will be stored
        file_name ([str]): file name of the current 3d model
    """
    # check whether the off file is valid
    # For some reason, there are some broken files in the original ModelNet40 dataset
    try:

        in_path = os.path.join(data_root, file_name)
        out_path = os.path.join(out_root, file_name)

        with open(in_path) as f:
            first_line = f.readline()
            # check whether the off file is valid
            if len(first_line) != 3:
                data = open(in_path, 'r').read().splitlines()
                data = pre_process(data=data)
                write_file(data=data,out_path=in_path)
    except Exception as e:
        LOGGER.error(" Error with check off file: {}".format(e))
        sys.exit(1)

    try:
        # compress to 1024 faces
        ms = ml.MeshSet()
        ms.load_new_mesh(in_path)
        ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=1024, preservenormal=True)
        ms.save_current_mesh(out_path)
    except Exception as e:
        LOGGER.error(" Error with compressing off file: {}".format(e))
        sys.exit(1)


def write_file(out_path, data):
    with open(out_path, 'w') as f:
        for x in data:
            f.write(str(x) + '\n')


def pre_process(data):
    data = deque(data)
    first_element = data.popleft()
    firstline, secondline= first_element[0:3], first_element[3:]
    
    data.appendleft(secondline)
    data.appendleft(firstline)
    data = list(data)
    return data


def main(data_root):
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "batch=", "path=", "filename="])
    for opt_name, opt_value in opts:
        if opt_name in ("-h", "--help"):
            print("Help yourself please :)")
        elif opt_name == "--batch":
            run_type = opt_value
        elif opt_name == "--filename":
            filename = opt_value
        elif opt_name == "--path":
            path = opt_value

    if run_type == "T":
        run_batch(data_root, LOAD_FEATURE_PATH)
    elif run_type == "F":
        if filename:
            run_single(path, SEARCH_FEATURE_PATH, filename)
        else:
            print("please provide filename to run a single file")
    else:
        print("T or F only you stupid")


def run_single(path, out_root, filename):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    process(data_root=path, out_root=out_root, file_name=filename)


def run_batch(data_root, out_root):
    print("start batch")
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    for file_name in os.listdir(data_root):
        print(file_name)
        if file_name.endswith("off"):
            process(data_root=data_root, out_root=out_root, file_name=file_name)


if __name__ == "__main__":
    main(DATA_PATH)