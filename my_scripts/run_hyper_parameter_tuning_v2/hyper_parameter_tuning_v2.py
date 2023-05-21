import pprint
import subprocess
import torch
from hyper_parameter_tuning_flexmatch_complete_runs import *
from hyper_parameter_tuning_flexmatch_test_runs import *
from hyper_parameter_tuning_freematch import *

if __name__ == "__main__":
    # version = 10
    # print("version - ",version)
    # call_adamatch_missing_labels(version)
    version = "10"
    print("version - ", version)
    # torch.cuda.empty_cache()
    flexmatch_missing_labels_cifar100_v53(version)
    flexmatch_missing_labels_cifar100_v54(version)
