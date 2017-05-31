import cPickle
import matplotlib.pyplot as plt
import math
import numpy as np

xlist = [i for i in range(20)] + [j*100 for j in range(40)]
xlist = np.array(xlist, dtype = np.float32)
xlist = np.log(xlist)

# nms = ["L2"]
nms = ["L12", "Lr2", "L2"]
# nms = ["L1", "L12_itlv_fine_ex_L1"]
# nms = ["L2", "L12_itlv_fine_ex", "L12"]
labels = ["Normal-Lb", "Noise-Lb", "Single-Lb"]
plt.figure()
annotation_dev=[-4,0.5]
for i in range(len(nms)):
    nm = nms[i]
    # name = "result_"+nm+".txtclean"
    name = "double_result_" + nm + ".txtclean"

    data = []
    with open(name, "r") as file:
        data = cPickle.load(file)

        
    flag_err = False
    if len(data) == 2:
        mean = data[0]
        std = data[1]
        length = len(mean)
        flag_err = True
        min_value = np.min(mean)
        min_index = mean.tolist().index(min_value)
        min_std = std[min_index]
        min_index = xlist[min_index]
    else:
        length = len(data)
        min_value = np.min(data)
        min_index = xlist[data.tolist().index(min_value)]

    if flag_err:
        plt.errorbar(xlist[:length], mean, std, label = labels[i])
        plt.legend()
        plt.annotate(("min_%s: %f +- %f" % (labels[i], min_value, min_std)), xy=(min_index, min_value),
                     xytext=(min_index+annotation_dev[0], min_value+annotation_dev[1]),
                     arrowprops=dict(facecolor='gray', shrink=1, width=0.1))
        for ax in range(2):
            annotation_dev[ax]+=0.5
    else:
        plt.plot(xlist[:length], data, label = labels[i])
plt.show()