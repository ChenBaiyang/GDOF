import os, time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model import GDOF


def get_labeled_anomalies(label, n_train_pos, seed):
    all_idx = np.arange(len(label))
    pos_idx = all_idx[label == 1]
    assert len(pos_idx) > n_train_pos
    np.random.seed(seed)
    np.random.shuffle(pos_idx)

    train_pos_idx = pos_idx[:n_train_pos]
    test_idx = np.setdiff1d(all_idx, train_pos_idx)

    return train_pos_idx, test_idx

model_name = 'GDOF'
result_file_name = f'results_{model_name}.csv'
open(result_file_name, 'w').write('dataset,alpha,n_neg,n_pos,seed,auc,pr\n')

dir = 'data/'
dataset_list = np.array(os.listdir(dir))

for dataset in dataset_list:
    d = np.load(dir+dataset)
    data = d['X']
    valids = np.std(data, axis=0) > 1e-6
    data = data[:, valids]
    n, m = data.shape
    label = d['y']
    nominals = d['nominals'][valids]
    print("{} Data shape:{}  # Outlier:{} # Nominals:{}".format(dataset[:-4], (n, m), label.sum(),nominals.sum()))

    for n_train in [10, 15, 20, 25, 30]:
        for seed in range(10):
            train_pos_idx, test_idx = get_labeled_anomalies(label, n_train, seed)

            for n_neg in [100]:
                t0 = time.time()
                model = GDOF(data, nominals)
                model.fit(pos_idx=train_pos_idx, n_neg=n_neg)
                out_scores = model.predict_score(test_idx, alpha=0.5)

                auc = roc_auc_score(label[test_idx], out_scores)
                pr = average_precision_score(y_true=label[test_idx], y_score=out_scores, pos_label=1)
                print(dataset[:-4], n_neg, n_train, seed, auc, pr, 'Time:', round(time.time()-t0, 1))

                scores = [dataset[:-4], str(0.5), str(n_neg), str(n_train), str(seed), str(auc)[:8], str(pr)[:8]]
                open(result_file_name, 'a').write(','.join(scores) + '\n')
                del model
            continue
        continue

