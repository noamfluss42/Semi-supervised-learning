import pprint
import subprocess
import torch
from hyper_parameter_tuning_flexmatch_complete_runs import *
from hyper_parameter_tuning_flexmatch_test_runs import *
from hyper_parameter_tuning_freematch import *
from hyper_parameter_tuning_add_missing_runs import *
from hyper_parameter_tuning_flexmatch_kl_divergence import *
from hyper_parameter_tuning_flexmatch_kl_divergence_freematch import *
if __name__ == "__main__":
    version = "10"
    print("version - ", version)
    # flexmatch_missing_labels_cifar100_v81_1(version)
    # flexmatch_missing_labels_cifar100_v81_2(version)
    # flexmatch_missing_labels_cifar100_v82(version)
    # flexmatch_missing_labels_cifar100_v84(version)
    # call_freematch_cifar100_missing_labels_v21(version)
    # call_freematch_cifar100_missing_labels_v22(version)
    #
    # call_freematch_cifar100_missing_labels_v23(version)
    # freematch_kl_divergence_missing_labels_cifar100_v3000(version)
    #
    # freematch_kl_divergence_missing_labels_cifar100_v3001(version)
    #
    # freematch_kl_divergence_missing_labels_cifar100_v3002(version)
    flexmatch_kl_divergence_missing_labels_cifar100_v10018(version)
    # flexmatch_kl_divergence_missing_labels_cifar100_v10001(version)
    # call_freematch_cifar100_missing_labels_v22(version)
