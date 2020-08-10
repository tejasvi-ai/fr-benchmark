import torch
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed
import random, pickle
import numpy as np


def shuffle_dict(d):
    l = list(d.items())
    random.shuffle(l)
    d = dict(l)
    for k in d:
        random.shuffle(d[k])
    return d


def load(fp):
    fp = "dlib_embeddings/" + fp
    with open(fp, "rb") as f:
        return pickle.load(f, encoding="bytes")


milpitas_embeddings = shuffle_dict(load("milpitas_nested_dlib.pkl"))
new_flex_milpitas_nested_embeddings = shuffle_dict(
    load("new_flex_milpitas_nested_dlib.pkl")
)
celeb_embeddings = shuffle_dict(load("celeb_dlib.pkl"))

unknown_ids = set(milpitas_embeddings.keys()) - set(
    new_flex_milpitas_nested_embeddings.keys()
)

# Processing with thresh=0.7..##########
#  19638  match out of 20719  tp_percent: 94.78256672619335
# 34  not_found out of 20753
#  855  match out of 1000  fp_percent: 85.5
# Processing with thresh=0.6..##########
#  19638  match out of 20717  tp_percent: 94.79171694743448
# 36  not_found out of 20753
#  228  match out of 1000  fp_percent: 22.8
# Processing with thresh=0.5..##########
#  19576  match out of 20586  tp_percent: 95.09375303604392
# 167  not_found out of 20753
#  24  match out of 1000  fp_percent: 2.4
# Processing with thresh=0.4..##########
#  18223  match out of 18774  tp_percent: 97.06509001811015
# 1979  not_found out of 20753
#  0  match out of 1000  fp_percent: 0.0
# Processing with thresh=0.3..##########
# 10143  match out of 10226  tp_percent: 99.18834343829455
# 10527  not_found out of 2075
# 0  match out of 1000  fp_percent: 0.0


fp_percent = tp_percent = total = tp = fp = not_found = 0
fp_paths = []


def process_true_positive(thresh=0.8):
    print(f"Processing true positives with thresh={thresh}.." + 10 * "#")
    global tp, tp_percent, total
    tp = tp_percent = total = not_found = 0

    for j, (tid, tembds) in enumerate(milpitas_embeddings.items()):
        print(
            f"\r[{j/len(milpitas_embeddings)*100:.2f}% samples processed] [{tp / (total -not_found + 1e-10) * 100:.2f}% True positive] [{not_found / (total+1e-10)*100:.2f}% Not found]",
            end="",
        )
        if tid in unknown_ids:
            continue
        for tembd, tpath in tembds:
            min_dist = float("inf")
            min_dist_id = None
            min_dit_rpath = None

            for rid, rembds in new_flex_milpitas_nested_embeddings.items():
                for rembd, rpath in rembds:
                    try:
                        dist = np.linalg.norm(rembd[0] - tembd[0])
                    except IndexError:
                        continue
                    if dist < thresh and dist < min_dist:
                        min_dist = dist
                        min_dist_id = rid
                        min_dit_rpath = rpath

            total += 1
            if min_dist_id is not None:
                if min_dist_id == tid:
                    tp += 1
            else:
                not_found += 1
    tp_percent = tp / (total - not_found) * 100
    print("\n", tp, " match out of", total - not_found, f" tp_percent: {tp_percent}")
    print(not_found, " not_found out of", total)


def process_false_positive(thresh=0.8):
    print(f"Processing false positives with thresh={thresh}..")
    global fp, fp_paths, fp_percent, total
    fp = fp_percent = total = 0
    fp_paths = []

    for j, (tid, tembds) in enumerate(celeb_embeddings.items()):
        for i, (tembd, tpath) in enumerate(tembds):
            print(
                f"\r[{i/len(tembds)*100:.2f}% samples processed] [{fp / (total + 1e-10) * 100:.2f}% False positive]",
                end="",
            )
            min_dist = float("inf")
            min_dist_id = None
            min_dit_rpath = None

            for rid, rembds in new_flex_milpitas_nested_embeddings.items():
                for rembd, rpath in rembds:
                    try:
                        dist = np.linalg.norm(rembd[0] - tembd[0])
                    except IndexError:
                        continue
                    if dist < thresh and dist < min_dist:
                        min_dist = dist
                        min_dist_id = rid
                        min_dit_rpath = rpath

            total += 1
            if min_dist_id is not None:
                fp += 1
                fp_paths.append((tpath, rpath))

    fp_percent = fp / total * 100
    print("\n", fp, " match out of", total, f" fp_percent: {fp_percent}")


for thresh in [0.4, 0.3]:
    process_true_positive(thresh=thresh)
    process_false_positive(thresh=thresh)

embed()
