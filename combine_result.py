import numpy as np
import matplotlib.pyplot as plt
import cPickle

dir_rn = "./random_block_result/"
dir_single = "./result/"

name = "L12_itlv_fine_ex"
# name = "result_" + name + ".txt"
name = "double_result_" + name + ".txt"


name_rn = dir_rn + name
name_single = dir_single + name

results = []

with open(name_single, "r") as file_single:
    data = []
    for line in file_single.readlines():
        if len(line)<=1 or "random_seed" in line.split(" ")[0] or "valid" not in line.split(" ")[0]:
            if len(data)>0:
                results.append(data)
                data = []
            continue
        num = line.split(" ")[-1][:7]
        print num
        data.append(float(num))
    if len(data) > 0:
        results.append(data)

try:
    with open(name_rn, "r") as file_rn:
        data = []
        for line in file_rn.readlines():
            if len(line)<=1 or "random_seed" in line.split(" ")[0] or "valid" not in line.split(" ")[0]:
                if len(data)>0:
                    results.append(data)
                    data = []
                continue
            num = line.split(" ")[-1][:7]
            print num
            data.append(float(num))
        if len(data) > 0:
            results.append(data)
except:
    print "error in openning random_block_result"

max_len = 0
for result in results:
    if len(result) > max_len:
        max_len = len(result)
for i in range(len(results)):
    len_i = len(results[i])
    for j in range(len_i, max_len):
        results[i].append(np.nan)
results = np.array(results)
mean = np.nanmean(results, axis = 0)
std = np.nanstd(results, axis = 0)

with open(name+"clean","w") as resultfile:
    cPickle.dump([mean,std], resultfile)