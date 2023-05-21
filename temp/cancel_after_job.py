import os.path
from os import listdir
from os.path import isfile, join
import subprocess

import numpy as np


# 15541941
def get_id(sqn_line):
    return sqn_line[sqn_line.find("1"):sqn_line.find("_")]

def main():
    sqn_result = subprocess.getoutput("squeue -u noam.fluss")
    sqn_result_lines = sqn_result.split("\n")[1:]
    for sqn_line in sqn_result_lines:
        print(f"sqn line {sqn_line},sqn job id {get_id(sqn_line)}")
        print(int(get_id(sqn_line)))
        if int(get_id(sqn_line)) > 15541941:
            subprocess.call(f"scancel {get_id(sqn_line)}", shell=True)


if __name__ == "__main__":
    main()
