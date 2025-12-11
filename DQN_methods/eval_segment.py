import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed

def compute_correlation(input_data, endmember_spectrum):
    return pearsonr(input_data, endmember_spectrum)[0]
import time
def selection_to_acc(data_path,endmember_path,selected_bands):
    data = np.load(data_path)['data_nm']
    label = np.load(data_path)['label']
    endmember = np.load(endmember_path)['data']
    test_data = data[88:].reshape(28, -1, 81)
    test_label = label[88:].reshape(28, -1).astype(np.int32)
    # print(test_data.shape)
    # print(test_label.shape)
    # print(endmember.shape)

    all_output_label = np.zeros_like(test_label)

    group_acc_record=np.zeros(9)

    #####在计算之前我们统计testlabel中本身包含的各种label的数量先
    stat_label = np.zeros(9)
    n,s = test_label.shape
    for i in range(n):
        for j in range(s):
            stat_label[test_label[i][j]]+=1
    print("原本的label数量统计")
    print(stat_label)

    #######
    n,s,c = test_data.shape

    def process_pixel(i, s_n):
        # print(i,s_n)
        input_data = test_data[i][s_n][selected_bands[i]]
        h_l = [0]
        for j in range(8):
            c = compute_correlation(input_data, endmember[j][selected_bands[i]])
            h_l.append(c)
        if np.max(h_l) < 0.7:
            h_l[0] = 1
        return np.argmax(h_l), test_label[i][s_n]

    results = Parallel(n_jobs=4)(delayed(process_pixel)(i, s_n) for i in range(n) for s_n in range(s))

    for index, (output_label, gt_label) in enumerate(results):
        i, s_n = divmod(index, s)
        all_output_label[i][s_n] = output_label
        if output_label == gt_label:
            group_acc_record[gt_label] += 1

    print("各个标签下的正确数量")
    print(group_acc_record)
    print("各个标签下的正确率为")
    group_acc_nm = [group_acc_record[i] / stat_label[i] for i in range(9)]
    print(group_acc_nm)

    print("AA均值报告:{} 方差为{}".format(np.mean(group_acc_nm), np.std(group_acc_nm)))
    print("OA均值报告:{}".format(np.sum(group_acc_record) / np.sum(stat_label)))



    test_label = test_label.reshape(-1)
    all_output_label = all_output_label.reshape(-1)
    conf = confusion_matrix(test_label, all_output_label)
    print(conf)

    kappa = cohen_kappa_score(test_label, all_output_label)
    print(kappa)

if __name__ == '__main__':
    selected_bands = [35, 31, 37, 29, 30, 33, 39, 34, 40, 32, 24, 41, 25, 26, 51, 23, 27, 6, 10, 28, 75, 36, 43, 71, 73, 63, 70, 8, 15, 52, 74, 14, 38, 18, 22, 69, 76, 53, 79, 49] #DA 40 nmdata
    # selected_bands = [0, 5, 9, 10, 16, 21, 24, 28, 29, 32, 35, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 68, 72, 75, 76] #BO  no 40 nmdata
    # selected_bands =[0, 18, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 72] #BO 40nmdata

    # selected_bands = [5, 12, 16, 18, 21, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 63, 65, 72, 76, 77, 79, 80]#BO no 40 nmdata
    selected_bands = [selected_bands for _ in range(28)]
    data_path = '../data/micro_data_0506/micro0507_200.npz'
    endmember_path = '../data/micro_data_0506/micro_standard.npz'
    time1 = time.time()
    selection_to_acc(data_path,endmember_path,selected_bands)
    print((time.time()-time1)/60)