import os
import gc
import glob
import json
import itertools
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import joblib

warnings.filterwarnings("ignore")

class MABe:
    # Đường dẫn data competition
    train_path = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
    test_path = "/kaggle/input/MABe-mouse-behavior-detection/test.csv"
    train_annotation_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation"
    train_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
    test_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"

    model_root = "/kaggle/input/models"
    model_subdir = "lightgbm"        # folder con trong model_root

    # "validate"  -> train + tìm threshold + lưu model/threshold
    # "submit"    -> chỉ load model có sẵn và tạo submission
    mode = "submit"

    # Tham số LightGBM (CPU)
    lgb_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )


train = pd.read_csv(MABe.train_path)
test = pd.read_csv(MABe.test_path)

# List các kiểu body parts
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

#drop các phần body parts khi quá nhiều
drop_body_parts = [
    "headpiece_bottombackleft", "headpiece_bottombackright",
    "headpiece_bottomfrontleft", "headpiece_bottomfrontright",
    "headpiece_topbackleft", "headpiece_topbackright",
    "headpiece_topfrontleft", "headpiece_topfrontright",
    "spine_1", "spine_2",
    "tail_middle_1", "tail_middle_2", "tail_midpoint"
]


#Quy đổi window/lag tính theo số frame ở 30fps sang số frame tương ứng ở fps thực tế của video
def _scale(n_frames_at_30fps, fps, ref=30.0):
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

#Tương tự với scale nhưng giữ dấu (+/0) để dùng cho các offset (quá khứ/ tương lai)
def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

#Lấy fps thật của video
def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if "frames_per_second" in meta_df.columns and pd.notnull(meta_df["frames_per_second"]).any():
        return float(meta_df["frames_per_second"].iloc[0])
    vid = meta_df["video_id"].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

#Sinh dữ liệu theo single và pair mouse
def generate_mouse_data(
    dataset,
    traintest,                     # 'train' hoặc 'test'
    traintest_directory=None,
    generate_single=True,
    generate_pair=True
):
    """
    Sinh dữ liệu:
      - 'single': từng con chuột (agent = target = self)
      - 'pair'  : cặp chuột (agent != target)
    Trả về generator: (switch, data, meta, label / actions)
    """
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if lab_id.startswith("MABe22"):
            continue

        if type(row.behaviors_labeled) != str:
            continue

        video_id = row.video_id
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        # Nếu bodypart quá nhiều thì bỏ bớt
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")

        # Pivot (video_frame, (mouse_id, bodypart, (x,y)))
        pvid = vid.pivot(
            columns=["mouse_id", "bodypart"],
            index="video_frame",
            values=["x", "y"]
        )
        del vid
        gc.collect()

        # Đưa về dạng multiindex: bodypart -> coord -> mouse_id
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T

        # Đổi đơn vị sang cm
        pvid /= row.pix_per_cm_approx

        # Parse behaviors
        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(",") for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=["agent", "target", "action"])

        # Annotation (train mới có)
        if traintest == "train":
            try:
                annot = pd.read_parquet(
                    path.replace("train_tracking", "train_annotation")
                )
            except FileNotFoundError:
                continue

        # ---------------- SINGLE (agent = self) ----------------
        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(
                        vid_behaviors_subset.query("agent == @mouse_id_str").action
                    )
                    single_mouse = pvid.loc[:, mouse_id]
                    single_mouse_meta = pd.DataFrame({
                        "video_id": video_id,
                        "agent_id": mouse_id_str,
                        "target_id": "self",
                        "video_frame": single_mouse.index
                    })

                    if traintest == "train":
                        single_mouse_label = pd.DataFrame(
                            0.0,
                            columns=vid_agent_actions,
                            index=single_mouse.index
                        )
                        annot_subset = annot.query(
                            "(agent_id == @mouse_id) & (target_id == @mouse_id)"
                        )
                        for i in range(len(annot_subset)):
                            r = annot_subset.iloc[i]
                            single_mouse_label.loc[r["start_frame"]:r["stop_frame"], r.action] = 1.0
                        yield "single", single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        # test: chỉ trả về list action
                        yield "single", single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass

        # ---------------- PAIR (agent != target) ----------------
        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(
                    np.unique(pvid.columns.get_level_values("mouse_id")), 2
                ):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(
                        vid_behaviors_subset
                        .query("(agent == @agent_str) & (target == @target_str)")
                        .action
                    )
                    mouse_pair = pd.concat(
                        [pvid[agent], pvid[target]],
                        axis=1,
                        keys=["A", "B"]
                    )
                    mouse_pair_meta = pd.DataFrame({
                        "video_id": video_id,
                        "agent_id": agent_str,
                        "target_id": target_str,
                        "video_frame": mouse_pair.index
                    })

                    if traintest == "train":
                        mouse_pair_label = pd.DataFrame(
                            0.0,
                            columns=vid_agent_actions,
                            index=mouse_pair.index
                        )
                        annot_subset = annot.query(
                            "(agent_id == @agent) & (target_id == @target)"
                        )
                        for i in range(len(annot_subset)):
                            r = annot_subset.iloc[i]
                            mouse_pair_label.loc[r["start_frame"]:r["stop_frame"], r.action] = 1.0
                        yield "pair", mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        yield "pair", mouse_pair, mouse_pair_meta, vid_agent_actions


def add_curvature_features(X, cx, cy, fps):
    vx = cx.diff()
    vy = cy.diff()
    ax = vx.diff()
    ay = vy.diff()

    cross_prod = vx * ay - vy * ax
    v_mag = np.sqrt(vx**2 + vy**2)
    curv = np.abs(cross_prod) / (v_mag**3 + 1e-6)

    for w in [25, 50, 75]:
        ws = _scale(w, fps)
        X[f"curv_mean_{w}"] = curv.rolling(ws, min_periods=max(1, ws // 5)).mean()

    ang = np.arctan2(vy, vx)
    ang_diff = np.abs(ang.diff())
    w = 30
    ws = _scale(w, fps)
    X[f"turn_rate_{w}"] = ang_diff.rolling(ws, min_periods=max(1, ws // 5)).sum()
    return X


def add_multiscale_features(X, cx, cy, fps):
    speed = np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)
    scales = [20, 40, 60, 80]
    for s in scales:
        ws = _scale(s, fps)
        if len(speed) >= ws:
            X[f"sp_m{s}"] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f"sp_s{s}"] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f"sp_m{scales[0]}" in X.columns and f"sp_m{scales[-1]}" in X.columns:
        X["sp_ratio"] = X[f"sp_m{scales[0]}"] / (X[f"sp_m{scales[-1]}"] + 1e-6)
    return X


def add_state_features(X, cx, cy, fps):
    speed = np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()
    try:
        bins = [-np.inf, 0.5, 2.0, 5.0, np.inf]
        states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [20, 40, 60, 80]:
            ws = _scale(window, fps)
            if len(states) >= ws:
                for st in [0, 1, 2, 3]:
                    X[f"s{st}_{window}"] = (
                        (states == st).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 5)).mean()
                    )
                change = (states != states.shift(1)).astype(float)
                X[f"trans_{window}"] = change.rolling(ws, min_periods=max(1, ws // 5)).sum()
    except Exception:
        pass
    return X


def add_longrange_features(X, cx, cy, fps):
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(cx) >= ws:
            X[f"x_ml{window}"] = cx.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f"y_ml{window}"] = cy.rolling(ws, min_periods=max(5, ws // 6)).mean()

    speed = np.sqrt(cx.diff()**2 + cy.diff()**2) * float(fps)
    for span in [30, 60, 120]:
        s = _scale(span, fps)
        X[f"x_e{span}"] = cx.ewm(span=s, min_periods=1).mean()
        X[f"y_e{span}"] = cy.ewm(span=s, min_periods=1).mean()
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f"sp_pct{window}"] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)
    return X

def transform_single(single_mouse, body_parts_tracked, fps):
    """
    Feature cho 1 con chuột (single).
    """
    available = single_mouse.columns.get_level_values(0)

    # Khoảng cách bình phương giữa các bodypart
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available and p2 in available
    })
    X = X.reindex(
        columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)],
        copy=False
    )

    # Một số feature tốc độ / hình dạng
    if all(p in single_mouse.columns for p in ["ear_left", "ear_right", "tail_base"]):
        lag = _scale(10, fps)
        shifted = single_mouse[["ear_left", "ear_right", "tail_base"]].shift(lag)
        speeds = pd.DataFrame({
            "sp_lf": np.square(single_mouse["ear_left"] - shifted["ear_left"]).sum(axis=1, skipna=False),
            "sp_rt": np.square(single_mouse["ear_right"] - shifted["ear_right"]).sum(axis=1, skipna=False),
            "sp_lf2": np.square(single_mouse["ear_left"] - shifted["tail_base"]).sum(axis=1, skipna=False),
            "sp_rt2": np.square(single_mouse["ear_right"] - shifted["tail_base"]).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if "nose+tail_base" in X.columns and "ear_left+ear_right" in X.columns:
        X["elong"] = X["nose+tail_base"] / (X["ear_left+ear_right"] + 1e-6)

    if all(p in available for p in ["nose", "body_center", "tail_base"]):
        v1 = single_mouse["nose"] - single_mouse["body_center"] #tính vector
        v2 = single_mouse["tail_base"] - single_mouse["body_center"]
        X["body_ang"] = (v1["x"] * v2["x"] + v1["y"] * v2["y"]) / (
            np.sqrt(v1["x"]**2 + v1["y"]**2) * np.sqrt(v2["x"]**2 + v2["y"]**2) + 1e-6
        ) #tính cos (v1,v2)

    if "body_center" in available:
        cx = single_mouse["body_center"]["x"]
        cy = single_mouse["body_center"]["y"]

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f"cx_m{w}"] = cx.rolling(ws, **roll).mean()
            X[f"cy_m{w}"] = cy.rolling(ws, **roll).mean()
            X[f"cx_s{w}"] = cx.rolling(ws, **roll).std()
            X[f"cy_s{w}"] = cy.rolling(ws, **roll).std()
            X[f"x_rng{w}"] = cx.rolling(ws, **roll).max() - cx.rolling(ws, **roll).min()
            X[f"y_rng{w}"] = cy.rolling(ws, **roll).max() - cy.rolling(ws, **roll).min()
            X[f"disp{w}"] = np.sqrt(
                cx.diff().rolling(ws, min_periods=1).sum()**2 +
                cy.diff().rolling(ws, min_periods=1).sum()**2
            )
            X[f"act{w}"] = np.sqrt(
                cx.diff().rolling(ws, min_periods=1).var() +
                cy.diff().rolling(ws, min_periods=1).var()
            )

        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)

    if all(p in available for p in ["nose", "tail_base"]):
        nt_dist = np.sqrt(
            (single_mouse["nose"]["x"] - single_mouse["tail_base"]["x"])**2 +
            (single_mouse["nose"]["y"] - single_mouse["tail_base"]["y"])**2
        )
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f"nt_lg{lag}"] = nt_dist.shift(l)
            X[f"nt_df{lag}"] = nt_dist - nt_dist.shift(l)

    if all(p in available for p in ["ear_left", "ear_right"]):
        ear_d = np.sqrt(
            (single_mouse["ear_left"]["x"] - single_mouse["ear_right"]["x"])**2 +
            (single_mouse["ear_left"]["y"] - single_mouse["ear_right"]["y"])**2
        )
        for off in [-30, -20, -10, 10, 20, 30]:
            o = _scale_signed(off, fps)
            X[f"ear_o{off}"] = ear_d.shift(-o)
        w = _scale(30, fps)
        X["ear_con"] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    return X.astype(np.float32, copy=False)


def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    if "body_center" not in avail_A or "body_center" not in avail_B:
        return X

    rel_x = mouse_pair["A"]["body_center"]["x"] - mouse_pair["B"]["body_center"]["x"]
    rel_y = mouse_pair["A"]["body_center"]["y"] - mouse_pair["B"]["body_center"]["y"]
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    A_vx = mouse_pair["A"]["body_center"]["x"].diff()
    A_vy = mouse_pair["A"]["body_center"]["y"].diff()
    B_vx = mouse_pair["B"]["body_center"]["x"].diff()
    B_vy = mouse_pair["B"]["body_center"]["y"].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f"A_ld{window}"] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f"B_ld{window}"] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f"chase_{w}"] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f"sp_cor{window}"] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X


def transform_pair(mouse_pair, body_parts_tracked, fps):
    """
    Feature cho cặp chuột (pair).
    """
    avail_A = mouse_pair["A"].columns.get_level_values(0)
    avail_B = mouse_pair["B"].columns.get_level_values(0)

    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair["A"][p1] - mouse_pair["B"][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(
        columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)],
        copy=False
    ) #reindex đồng nhất schema feature giữa các video(đủ cột + đúng thứ tự) để model training không bị lệch cột hoặc lỗi thiếu feature

    # Thêm một số feature interaction
    if ("A", "ear_left") in mouse_pair.columns and ("B", "ear_left") in mouse_pair.columns:
        lag = _scale(10, fps)
        shA = mouse_pair["A"]["ear_left"].shift(lag)
        shB = mouse_pair["B"]["ear_left"].shift(lag)
        speeds = pd.DataFrame({
            "sp_A": np.square(mouse_pair["A"]["ear_left"] - shA).sum(axis=1, skipna=False),
            "sp_AB": np.square(mouse_pair["A"]["ear_left"] - shB).sum(axis=1, skipna=False),
            "sp_B": np.square(mouse_pair["B"]["ear_left"] - shB).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)
        
    if all(p in avail_A for p in ["nose", "tail_base"]) and all(p in avail_B for p in ["nose", "tail_base"]):
        dir_A = mouse_pair["A"]["nose"] - mouse_pair["A"]["tail_base"]
        dir_B = mouse_pair["B"]["nose"] - mouse_pair["B"]["tail_base"]
        X["rel_ori"] = (dir_A["x"] * dir_B["x"] + dir_A["y"] * dir_B["y"]) / (
            np.sqrt(dir_A["x"]**2 + dir_A["y"]**2) *
            np.sqrt(dir_B["x"]**2 + dir_B["y"]**2) + 1e-6
        )

    if all(p in avail_A for p in ["nose"]) and all(p in avail_B for p in ["nose"]):
        cur = np.square(mouse_pair["A"]["nose"] - mouse_pair["B"]["nose"]).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair["A"]["nose"].shift(lag)
        shB_n = mouse_pair["B"]["nose"].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X["appr"] = cur - past

    if "body_center" in avail_A and "body_center" in avail_B:
        cd = np.sqrt(
            (mouse_pair["A"]["body_center"]["x"] - mouse_pair["B"]["body_center"]["x"])**2 +
            (mouse_pair["A"]["body_center"]["y"] - mouse_pair["B"]["body_center"]["y"])**2
        )
        X["v_cls"] = (cd < 5.0).astype(float)
        X["cls"] = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X["med"] = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X["far"] = (cd >= 30.0).astype(float)

        cd_full = np.square(mouse_pair["A"]["body_center"] - mouse_pair["B"]["body_center"]).sum(axis=1, skipna=False)

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f"d_m{w}"] = cd_full.rolling(ws, **roll).mean()
            X[f"d_s{w}"] = cd_full.rolling(ws, **roll).std()
            X[f"d_mn{w}"] = cd_full.rolling(ws, **roll).min()
            X[f"d_mx{w}"] = cd_full.rolling(ws, **roll).max()

            d_var = cd_full.rolling(ws, **roll).var()
            X[f"int{w}"] = 1 / (1 + d_var)

            Axd = mouse_pair["A"]["body_center"]["x"].diff()
            Ayd = mouse_pair["A"]["body_center"]["y"].diff()
            Bxd = mouse_pair["B"]["body_center"]["x"].diff()
            Byd = mouse_pair["B"]["body_center"]["y"].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f"co_m{w}"] = coord.rolling(ws, **roll).mean()
            X[f"co_s{w}"] = coord.rolling(ws, **roll).std()

    if "nose" in avail_A and "nose" in avail_B:
        nn = np.sqrt(
            (mouse_pair["A"]["nose"]["x"] - mouse_pair["B"]["nose"]["x"])**2 +
            (mouse_pair["A"]["nose"]["y"] - mouse_pair["B"]["nose"]["y"])**2
        )
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f"nn_lg{lag}"] = nn.shift(l)
            X[f"nn_ch{lag}"] = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            X[f"cl_ps{lag}"] = is_cl.rolling(l, min_periods=1).mean()

    if "body_center" in avail_A and "body_center" in avail_B:
        Avx = mouse_pair["A"]["body_center"]["x"].diff()
        Avy = mouse_pair["A"]["body_center"]["y"].diff()
        Bvx = mouse_pair["B"]["body_center"]["x"].diff()
        Bvy = mouse_pair["B"]["body_center"]["y"].diff()
        val = (Avx * Bvx + Avy * Bvy) / (
            np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6
        )

        for off in [-30, -20, -10, 0, 10, 20, 30]:
            o = _scale_signed(off, fps)
            X[f"va_{off}"] = val.shift(-o)

        w = _scale(30, fps)
        cd_full = np.square(mouse_pair["A"]["body_center"] - mouse_pair["B"]["body_center"]).sum(axis=1, skipna=False)
        X["int_con"] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                       (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)


def simple_tune_threshold(oof, y):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.0, 1.0, 51):
        f1 = f1_score(y, (oof >= t), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def cross_validate_classifier(X, label, meta, body_parts_tracked_str, section, mode_key):
    os.makedirs(f"{MABe.model_subdir}/{section}", exist_ok=True)

    oof_df = pd.DataFrame(index=meta.video_frame)
    f1_list = []
    thresholds = {}
    submission_list = []

    for action in label.columns:
        mask = ~label[action].isna().values
        y_action = label[action][mask].values.astype(int)
        X_action = X[mask]
        groups = meta.video_id[mask]

        if len(np.unique(groups)) < 2:
            continue
        if (y_action == 0).all():
            oof_action = np.zeros(len(y_action))
            thresholds[action] = 0.5
        else:
            try:
                cv = GroupKFold(n_splits=min(2, len(np.unique(groups))))
                oof_action = np.zeros(len(y_action))
                models = []

                for tr_idx, va_idx in cv.split(X_action, y_action, groups):
                    model = LGBMClassifier(**MABe.lgb_params)
                    model.fit(X_action.iloc[tr_idx], y_action[tr_idx])
                    oof_action[va_idx] = model.predict_proba(X_action.iloc[va_idx])[:, 1]
                    models.append(model)

                thr, f1_val = simple_tune_threshold(oof_action, y_action)
                thresholds[action] = thr
                f1_list.append((body_parts_tracked_str, mode_key, action, f1_val, thr))
                print(f"\t[{mode_key}] Action={action}, F1={f1_val:.4f}, thr={thr:.3f}")

                # Lưu model
                model_dir = f"{MABe.model_subdir}/{section}/{action}"
                os.makedirs(model_dir, exist_ok=True)
                joblib.dump(models, f"{model_dir}/lgb_trainer.pkl")
            except Exception as e:
                print(f"\tFailed train action {action}: {e}")
                oof_action = np.zeros(len(y_action))
                thresholds[action] = 0.5

        tmp = np.zeros(len(label))
        tmp[mask] = oof_action
        oof_df[action] = tmp

        del oof_action, mask, X_action, y_action, groups
        gc.collect()

    sub_part = predict_multiclass(oof_df, meta, thresholds)
    submission_list.append(sub_part)
    return submission_list, f1_list, thresholds


def predict_multiclass(pred, meta, thresholds, min_duration=3, smoothing_window=5):
    if pred.empty:
        return pd.DataFrame()

    required_cols = {"video_id", "agent_id", "target_id", "video_frame"}
    if not required_cols.issubset(set(meta.columns)):
        print("\t    Meta missing required columns")
        return pd.DataFrame()

    # Áp dụng temporal smoothing
    if smoothing_window > 1 and len(pred) >= smoothing_window:
        pred_smooth = pred.rolling(
            window=smoothing_window, 
            center=True, 
            min_periods=1
        ).mean()
    else:
        pred_smooth = pred

    # Chọn action có prob max
    ama = np.argmax(pred_smooth.values, axis=1)
    max_proba = pred_smooth.max(axis=1).values

    threshold_array = np.array([thresholds.get(col, 0.27) for col in pred.columns])
    action_thresholds = threshold_array[ama]
    ama = np.where(max_proba >= action_thresholds, ama, -1)

    # Áp dụng majority voting trong window nhỏ để giảm noise
    voting_window = 3
    if len(ama) >= voting_window:
        ama_smooth = pd.Series(ama).rolling(
            window=voting_window, 
            center=True, 
            min_periods=1
        ).apply(lambda x: pd.Series(x).mode()[0] if len(pd.Series(x).mode()) > 0 else x.iloc[0])
        ama = ama_smooth.values.astype(int)

    submission_parts = []
    if len(ama) == 0:
        return pd.DataFrame()

    current_action = ama[0]
    start_frame = meta.video_frame.iloc[0]
    duration = 1

    for i in range(1, len(ama)):
        if ama[i] == current_action:
            duration += 1
            continue

        if current_action >= 0 and duration >= min_duration:
            try:
                video_id = meta.video_id.iloc[i - 1]
                agent_id = meta.agent_id.iloc[i - 1]
                target_id = meta.target_id.iloc[i - 1]
                stop_frame = meta.video_frame.iloc[i - 1] + 1
                submission_parts.append({
                    "video_id": video_id,
                    "agent_id": agent_id,
                    "target_id": target_id,
                    "action": pred.columns[current_action],
                    "start_frame": start_frame,
                    "stop_frame": stop_frame
                })
            except Exception as e:
                print(f"\t      Error creating segment: {e}")

        current_action = ama[i]
        start_frame = meta.video_frame.iloc[i]
        duration = 1

    # Đoạn cuối
    if current_action >= 0 and duration >= min_duration:
        try:
            video_id = meta.video_id.iloc[-1]
            agent_id = meta.agent_id.iloc[-1]
            target_id = meta.target_id.iloc[-1]
            stop_frame = meta.video_frame.iloc[-1] + 1
            submission_parts.append({
                "video_id": video_id,
                "agent_id": agent_id,
                "target_id": target_id,
                "action": pred.columns[current_action],
                "start_frame": start_frame,
                "stop_frame": stop_frame
            })
        except Exception as e:
            print(f"\t      Error creating last segment: {e}")

    return pd.DataFrame(submission_parts)

def predict_with_ensemble_weights(X_te, models, action, weight_strategy="uniform"):
    """
    Dự đoán với weighted ensemble
    weight_strategy: "uniform", "inverse_fold", "performance_based"
    """
    probas = []
    for mdl in models:
        prob = mdl.predict_proba(X_te)[:, 1]
        probas.append(prob)
    
    if weight_strategy == "uniform":
        return np.mean(probas, axis=0)
    elif weight_strategy == "inverse_fold":
        weights = np.arange(1, len(probas) + 1)
        weights = weights / weights.sum()
        return np.average(probas, axis=0, weights=weights)
    else:
        return np.mean(probas, axis=0)

def load_thresholds_for_submit():
    """
    Tìm thresholds.pkl trong:
        /kaggle/input/kk-lightgbm/thresholds.pkl
        /kaggle/input/kk-lightgbm/lightgbm/thresholds.pkl
    """
    candidates = [
        os.path.join(MABe.model_root, "thresholds.pkl"),
        os.path.join(MABe.model_root, MABe.model_subdir, "thresholds.pkl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"Loaded thresholds from: {p}")
            return joblib.load(p)

    print("WARNING: thresholds.pkl not found, using default 0.5")
    return {
        "single": {"default": 0.5},
        "pair": {"default": 0.5}
    }

def submit(body_parts_tracked_str, switch_tr, section, thresholds_section):
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    print(f"\tTest subset size: {len(test_subset)}")

    if len(test_subset) == 0:
        print("\tNo test videos for this body_parts_tracked.")
        return []

    fps_lookup = (
        test_subset[["video_id", "frames_per_second"]]
        .drop_duplicates("video_id")
        .set_index("video_id")["frames_per_second"]
        .to_dict()
    )

    models_cache = {}
    all_actions = set()

    for _, row in test_subset.iterrows():
        if type(row.behaviors_labeled) == str:
            try:
                behaviors = json.loads(row.behaviors_labeled)
                for b in behaviors:
                    if isinstance(b, str) and "," in b:
                        parts = b.split(",")
                        action = parts[2] if len(parts) >= 3 else b
                    else:
                        action = str(b)
                    all_actions.add(action)
            except Exception as e:
                print(f"\tFailed to parse behaviors: {e}")

    print(f"\tWill try to load models for {len(all_actions)} actions: {list(all_actions)}")

    loaded_actions = 0
    for action in all_actions:
        try:
            pattern1 = os.path.join(MABe.model_root, MABe.model_subdir, str(section), action, "lgb_trainer.pkl")
            pattern2 = os.path.join(MABe.model_root, str(section), action, "lgb_trainer.pkl")

            model_files = glob.glob(pattern1)
            if not model_files:
                model_files = glob.glob(pattern2)

            if model_files:
                models = joblib.load(model_files[0])
                models_cache[action] = models
                loaded_actions += 1
                print(f"\t✓ Loaded model for action '{action}' from {model_files[0]}")
            else:
                print(f"\t✗ Model not found for action '{action}'")
        except Exception as e:
            print(f"\t✗ Failed to load model for action '{action}': {e}")

    print(f"\tLoaded {loaded_actions}/{len(all_actions)} actions.")
    if loaded_actions == 0:
        print("\tNo models loaded, skip this section.")
        return []

    def process_single_video(row):
        try:
            lab_id = row["lab_id"]
            video_id = row["video_id"]
            print(f"\tProcessing video: {video_id}")

            video_data = pd.DataFrame([row])
            gen = generate_mouse_data(
                video_data,
                "test",
                traintest_directory=MABe.test_tracking_path,
                generate_single=(switch_tr == "single"),
                generate_pair=(switch_tr == "pair")
            )

            video_submissions = []
            count_items = 0

            for sw_te, data_te, meta_te, actions_te in gen:
                count_items += 1
                print(f"\t  Part: {sw_te}, actions: {list(actions_te)}")

                if "video_frame" not in meta_te.columns:
                    print("\t  meta_te missing video_frame")
                    continue

                fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
                if sw_te == "single":
                    X_te = transform_single(data_te, body_parts_tracked, fps_i).astype(np.float32)
                else:
                    X_te = transform_pair(data_te, body_parts_tracked, fps_i).astype(np.float32)

                del data_te
                gc.collect()

                if X_te.shape[0] == 0:
                    print("\t  Warning: empty feature matrix")
                    continue

                pred = pd.DataFrame(index=meta_te.video_frame)
                predicted_actions = 0

                for action in actions_te:
                    if action not in models_cache:
                        continue
                    try:
                        models = models_cache[action]
                        mean_prob = predict_with_ensemble_weights(
                            X_te, models, action, weight_strategy="uniform"
                        )
                        pred[action] = mean_prob
                        predicted_actions += 1
                        thr = thresholds_section.get(action, 0.27)
                        print(f"\t    {action}: mean={mean_prob.mean():.4f}, max={mean_prob.max():.4f}, thr={thr}")
                    except Exception as e:
                        print(f"\t    Failed to predict action '{action}': {e}")

                del X_te
                gc.collect()

                print(f"\t    Predicted {predicted_actions}/{len(actions_te)} actions.")

                if not pred.empty and pred.shape[1] > 0:
                    # SỬ DỤNG PREDICT CÓ SMOOTHING
                    part_sub = predict_multiclass(
                        pred, meta_te, thresholds_section, 
                        min_duration=2, smoothing_window=5
                    )
                    if len(part_sub) > 0:
                        video_submissions.append(part_sub)
                        print(f"\t    Got {len(part_sub)} segments from this part.")
                    else:
                        print("\t    All segments filtered by thresholds.")
                else:
                    print("\t    No valid predictions for this part.")

                del pred
                gc.collect()

            print(f"\tVideo {video_id} done: {len(video_submissions)} partial submissions, {count_items} generator items.")
            return video_submissions
        except Exception as e:
            print(f"\tError processing video {row['video_id']}: {e}")
            return []

    submission_list = []
    print(f"\tStart processing {len(test_subset)} videos for switch_tr={switch_tr} ...")
    for idx, row in test_subset.iterrows():
        print(f"\n\t=== Video {idx + 1}/{len(test_subset)} ===")
        result = process_single_video(row)
        if result:
            submission_list.extend(result)
        if idx % 5 == 0:
            gc.collect()

    del models_cache
    gc.collect()

    print(f"\tFinished section={section}, switch_tr={switch_tr}, got {len(submission_list)} partial submissions.")
    return submission_list


def post_process_segments(submission_df, min_gap=5, min_duration=3):
    """
    Merge các segments gần nhau và loại bỏ segments quá ngắn
    min_gap: số frame tối thiểu giữa 2 segments để coi là riêng biệt
    min_duration: độ dài tối thiểu của segment
    """
    if len(submission_df) == 0:
        return submission_df
    
    processed_list = []
    
    for (video_id, agent_id, target_id, action), group in submission_df.groupby(
        ["video_id", "agent_id", "target_id", "action"]
    ):
        group = group.sort_values("start_frame").reset_index(drop=True)
        
        if len(group) == 0:
            continue
            
        merged = []
        current_start = group.iloc[0]["start_frame"]
        current_stop = group.iloc[0]["stop_frame"]
        
        for i in range(1, len(group)):
            row = group.iloc[i]
            # Nếu gap nhỏ hơn threshold thì merge
            if row["start_frame"] - current_stop <= min_gap:
                current_stop = max(current_stop, row["stop_frame"])
            else:
                # Lưu segment hiện tại nếu đủ dài
                if current_stop - current_start >= min_duration:
                    merged.append({
                        "video_id": video_id,
                        "agent_id": agent_id,
                        "target_id": target_id,
                        "action": action,
                        "start_frame": current_start,
                        "stop_frame": current_stop
                    })
                current_start = row["start_frame"]
                current_stop = row["stop_frame"]
        
        # Xử lý segment cuối cùng
        if current_stop - current_start >= min_duration:
            merged.append({
                "video_id": video_id,
                "agent_id": agent_id,
                "target_id": target_id,
                "action": action,
                "start_frame": current_start,
                "stop_frame": current_stop
            })
        
        processed_list.extend(merged)
    
    return pd.DataFrame(processed_list)

def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    old_submission = submission.copy()

    # Đảm bảo start/stop là int
    submission["start_frame"] = pd.to_numeric(
        submission["start_frame"], errors="coerce"
    ).fillna(0).astype(int)
    submission["stop_frame"] = pd.to_numeric(
        submission["stop_frame"], errors="coerce"
    ).fillna(0).astype(int)

    # Bỏ dòng start >= stop
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("NOTE: Dropped rows with start_frame >= stop_frame")

    old_submission = submission.copy()
    group_list = []

    # Xử lý trùng lặp theo (video, agent, target)
    for _, group in submission.groupby(["video_id", "agent_id", "target_id"]):
        group = group.sort_values("start_frame")
        mask = np.ones(len(group), dtype=bool)
        last_stop = -1
        for i, (_, row) in enumerate(group.iterrows()):
            if row["start_frame"] < last_stop:
                mask[i] = False
            else:
                last_stop = row["stop_frame"]
        group_list.append(group[mask])

    submission = pd.concat(group_list)
    if len(submission) != len(old_submission):
        print("NOTE: Dropped overlapping segments")

    # Điền thêm video không có prediction
    s_list = []
    for _, row in dataset.iterrows():
        lab_id = row["lab_id"]
        if lab_id.startswith("MABe22"):
            continue

        video_id = row["video_id"]
        if (submission.video_id == video_id).any():
            continue
        if type(row.behaviors_labeled) != str:
            continue

        print(f"Video {video_id} has no predictions -> filling dummy segments")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = json.loads(row["behaviors_labeled"])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(",") for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=["agent", "target", "action"])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(["agent", "target"]):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, ar) in enumerate(actions.iterrows()):
                b_start = start_frame + i * batch_length
                b_stop = min(b_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, ar["action"], b_start, b_stop))

    if len(s_list) > 0:
        df_fill = pd.DataFrame(
            s_list,
            columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]
        )
        submission = pd.concat([submission, df_fill], ignore_index=True)
        print("NOTE: Filled empty videos with dummy segments")

    submission = submission.reset_index(drop=True)
    return submission


submission_list = []
f1_list = []

if MABe.mode == "validate":
    thresholds_all = {"single": {}, "pair": {}}

    for section in range(1, len(body_parts_tracked_list)):
        body_parts_tracked_str = body_parts_tracked_list[section]
        try:
            body_parts_tracked = json.loads(body_parts_tracked_str)
            print(f"\n[{section}/{len(body_parts_tracked_list)-1}] body_parts_tracked = {body_parts_tracked}")

            if len(body_parts_tracked) > 5:
                body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

            train_subset = train[train.body_parts_tracked == body_parts_tracked_str]
            if len(train_subset) == 0:
                print("\tNo train videos for this subset.")
                continue

            fps_lookup = (
                train_subset[["video_id", "frames_per_second"]]
                .drop_duplicates("video_id")
                .set_index("video_id")["frames_per_second"]
                .to_dict()
            )

            single_data, single_meta, single_label = [], [], []
            pair_data, pair_meta, pair_label = [], [], []

            # Sinh dữ liệu từ train
            for sw, data, meta, label in generate_mouse_data(train_subset, "train"):
                if sw == "single":
                    single_data.append(data)
                    single_meta.append(meta)
                    single_label.append(label)
                else:
                    pair_data.append(data)
                    pair_meta.append(meta)
                    pair_label.append(label)
                del data, meta, label
            gc.collect()

            # -------- TRAIN CHO SINGLE --------
            if len(single_data) > 0:
                feats = []
                for d, m in zip(single_data, single_meta):
                    fps_i = _fps_from_meta(m, fps_lookup, default_fps=30.0)
                    X_i = transform_single(d, body_parts_tracked, fps_i).astype(np.float32)
                    feats.append(X_i)
                    del X_i, fps_i
                gc.collect()

                X_tr = pd.concat(feats, axis=0, ignore_index=True)
                y_tr = pd.concat(single_label, axis=0, ignore_index=True)
                meta_tr = pd.concat(single_meta, axis=0, ignore_index=True)

                tmp_sub_list, tmp_f1, tmp_thr = cross_validate_classifier(
                    X_tr, y_tr, meta_tr, body_parts_tracked_str, section, mode_key="single"
                )
                thresholds_all["single"][str(section)] = tmp_thr
                f1_list.extend(tmp_f1)
                submission_list.extend(tmp_sub_list)
                del feats, single_data, single_label, single_meta, X_tr, y_tr, meta_tr
                gc.collect()

            # -------- TRAIN CHO PAIR --------
            if len(pair_data) > 0:
                feats = []
                for d, m in zip(pair_data, pair_meta):
                    fps_i = _fps_from_meta(m, fps_lookup, default_fps=30.0)
                    X_i = transform_pair(d, body_parts_tracked, fps_i).astype(np.float32)
                    feats.append(X_i)
                    del X_i, fps_i
                gc.collect()

                X_tr = pd.concat(feats, axis=0, ignore_index=True)
                y_tr = pd.concat(pair_label, axis=0, ignore_index=True)
                meta_tr = pd.concat(pair_meta, axis=0, ignore_index=True)

                tmp_sub_list, tmp_f1, tmp_thr = cross_validate_classifier(
                    X_tr, y_tr, meta_tr, body_parts_tracked_str, section, mode_key="pair"
                )
                thresholds_all["pair"][str(section)] = tmp_thr
                f1_list.extend(tmp_f1)
                submission_list.extend(tmp_sub_list)
                del feats, pair_data, pair_label, pair_meta, X_tr, y_tr, meta_tr
                gc.collect()

        except Exception as e:
            print(f"\tError in section {section}: {e}")

    if len(f1_list) > 0:
        f1_df = pd.DataFrame(
            f1_list,
            columns=["body_parts_tracked_str", "mode", "action", "binary_F1", "threshold"]
        )
        print(f"\nMean F1 (binary): {f1_df['binary_F1'].mean():.4f}")
    else:
        f1_df = pd.DataFrame(columns=["body_parts_tracked_str", "mode", "action", "binary_F1", "threshold"])
        print("No F1 data collected.")

    os.makedirs(MABe.model_subdir, exist_ok=True)
    joblib.dump(thresholds_all, os.path.join(MABe.model_subdir, "thresholds.pkl"))
    joblib.dump(f1_df, os.path.join(MABe.model_subdir, "scores.pkl"))
    print(f"\nSaved thresholds & scores into folder: {MABe.model_subdir}/")

elif MABe.mode == "submit":
    # Dùng model + threshold đã có sẵn trong /kaggle/input/kk-lightgbm
    thresholds_all = load_thresholds_for_submit()

    for section in range(1, len(body_parts_tracked_list)):
        body_parts_tracked_str = body_parts_tracked_list[section]
        try:
            body_parts_tracked = json.loads(body_parts_tracked_str)
        except Exception:
            continue

        print(f"\n[{section}/{len(body_parts_tracked_list)-1}] body_parts_tracked = {body_parts_tracked}")

        # single
        thr_single = thresholds_all.get("single", {}).get(str(section), {})
        if thr_single:
            submission_list.extend(
                submit(body_parts_tracked_str, "single", section, thr_single)
            )

        # pair
        thr_pair = thresholds_all.get("pair", {}).get(str(section), {})
        if thr_pair:
            submission_list.extend(
                submit(body_parts_tracked_str, "pair", section, thr_pair)
            )

    if len(submission_list) > 0:
        submission = pd.concat(submission_list, ignore_index=True)
        print("\nApplying post-processing...")
        submission = post_process_segments(submission, min_gap=5, min_duration=2)
        print(f"After post-processing: {len(submission)} segments")
    else:
        # Fallback: 1 dòng dummy cho chắc
        submission = pd.DataFrame({
            "video_id": [test.video_id.iloc[0]],
            "agent_id": ["mouse1"],
            "target_id": ["self"],
            "action": ["rear"],
            "start_frame": [0],
            "stop_frame": [100]
        })

    submission_robust = robustify(submission, test, "test")
    submission_robust.index.name = "row_id"
    submission_robust.to_csv("submission.csv", index=True)
    print(f"\nSaved submission.csv, shape = {submission_robust.shape}")
    print(submission_robust.head())

