"""drift_check.py — chuỗi LLP túi 32 trên breast_cancer với mini-TabPFN: khởi tạo từ nhãn THẬT, khởi động ấm,
và ngẫu nhiên; ghi độ chính xác của trạng thái chuỗi so với nhãn thật sau mỗi vòng quét. Nếu chuỗi khởi tạo từ
nhãn thật vẫn trôi xuống, vấn đề là phân phối dừng của chuỗi xấp xỉ (sai số của q bị khuếch đại — Định lý 2),
không phải khởi tạo."""
import os, sys, json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, "/home/claude/mini_tabpfn")
from icgibbs import LLP, InContextGibbs, SklearnTFMConditional, make_bags
from icgibbs.baselines import warm_start_llp
from inference import MiniTabPFNClassifier
est1 = MiniTabPFNClassifier("/home/claude/runs/clf.pt", n_estimators=1, preprocess=("quantile",), max_features=10, temperature=1.0, device="cpu")
cond = SklearnTFMConditional(lambda: est1, 2)
X, y = load_breast_cancer(return_X_y=True); y = 1 - y; X = X.astype(np.float32)
out = {}
for bag in (32, 8):
    rng = np.random.default_rng(0)
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
    Xtr, ytr = X[itr], y[itr]
    bags, counts = make_bags(len(ytr), bag, rng, ytr)
    P0 = warm_start_llp(Xtr, bags, counts, seed=0)
    for name, init in (("true", ytr.copy()), ("warm", P0), ("lik", "lik")):
        g = InContextGibbs(cond, LLP(bags, counts, len(ytr)), init=init, n_blocks=10, n_sweeps=40, burn_in=40,
                           seed=0, monitor=lambda yy: (yy == ytr).mean()).run(Xtr)
        out[f"bag{bag}_{name}"] = [round(v, 3) for v in g.trace_.monitor]
        print(bag, name, out[f"bag{bag}_{name}"][::4], flush=True)
json.dump(out, open(os.path.join(HERE, "..", "..", "results", "pilot_v1_followup", "drift_check.json"), "w"), indent=1)
