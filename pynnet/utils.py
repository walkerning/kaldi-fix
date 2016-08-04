
import numpy as np

def prune_mat(mat, sparsity):
    flatten_data = mat.flatten()
    rank = np.argsort(abs(flatten_data))
    flatten_data[rank[:-int(rank.size * sparsity)]] = 0
    flatten_data = flatten_data.reshape(mat.shape)
    np.copyto(mat, flatten_data)

def redensify_mat(mat):
    std = 5e-6
    mat[:] += ((np.random.randint(0,2, size=mat.size)*2-1) * std).reshape(mat.shape)

def calc_prune_ratio(mats, sparsities):
    mat_sizes = np.array(map(lambda x:x.size, mats))
    sparsities = np.array(sparsities)
    return np.sum(mat_sizes * sparsities) / np.sum(mat_sizes)

def get_prune_sparsities(mats, names, wer, original_wer, sparsities = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75]):
    sparsities += [1]
    output_number = len(sparsities)
    mat_sizes = np.array(map(lambda x:x.size, mats))
    sparsities = np.array(sparsities)
    wer_all = np.zeros((wer.shape[0], wer.shape[1]+1))

    wer_all[:,:-1] = wer
    wer_all[:,-1] = original_wer

    wer_deri_avg = (wer_all[:,:-1] - wer_all[:,-1:]) / (sparsities[-1] - sparsities[:-1])
    wer_deri_weighted = wer_deri_avg / mat_sizes[:,None]

    med_wer_deri = np.median(wer_deri_weighted)

    threshs = np.arange(1,1+output_number).astype('f') / output_number * 2 * med_wer_deri
    for thresh in threshs:
        ratio = []
        error_inc = 0
        for mat_id in range(len(mat_sizes)):
            ids = np.where(wer_deri_weighted[mat_id] < thresh)[0]
            i = np.min(ids) if len(ids) > 0 else len(sparsities)-1
            ratio.append(sparsities[i])

            error_inc += wer_all[mat_id,i] - wer_all[mat_id, -1]

        print "======================================="
        print ("%.2f, "*len(ratio))%tuple(ratio)
        print "Error increased(estimated): ", error_inc
