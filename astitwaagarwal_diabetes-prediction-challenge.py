import numpy as np
import pandas as pd

import os, ast, shutil, copy

from bokeh.plotting import figure, gridplot
from bokeh.io import output_file, show, output_notebook
output_notebook()


def bokeh_show(
        params,
        df_cross,
        show_figures1,
        show_figures2, wps_fig2,
        color_cross):

    colors = [subm["color"] for subm in params["subm"]]

    def load_orders():
        desc = pd.read_csv("tida_desc.csv")
        return [ast.literal_eval(str(row.alls)) for row in desc.itertuples()]

    def build_dossiers(order_rows):
        subm_names = sorted(order_rows[0])
        cols = [[row[i] for row in order_rows] for i in range(len(subm_names))]

        def dossier(idx):
            def count_at(col_idx):
                return sum(1 for item in cols[col_idx] if item == subm_names[idx])
            return {
                "name": subm_names[idx],
                "q_in": [{"c": j, "q": count_at(j)} for j in range(len(subm_names))]
            }

        dossiers = [dossier(i) for i in range(len(subm_names))]
        return subm_names, dossiers, cols

    order_rows = load_orders()
    subm_names, dossiers, cols = build_dossiers(order_rows)

    figures1, qss = [], []
    height = 100 if len(colors) == 2 else 134 if len(colors) == 3 else (154 if len(colors) == 4 else 174)
    width_map = {5: 157, 4: 140, 8: 121, 9: 131, 10: 141, 11: 171}

    for idx, one_dossier in enumerate(dossiers):
        i_col = f"alls. {one_dossier['q_in'][idx]['c']}"
        qs = [one["q"] for one in one_dossier["q_in"]]
        x_names = [name.replace("Group", "").replace("subm_", "") for name in subm_names]
        width = width_map.get(len(colors), 130)
        f = figure(x_range=x_names, width=width, height=height, title=i_col)
        f.vbar(x=x_names, width=0.585, top=qs, color=colors)
        figures1.append(f)
        qss.append(qs)

    grid = gridplot([figures1])
    output_file("tida_alls.html")
    if show_figures1:
        show(grid)

    sub_wts = params["subwts"]
    main_wts = [subm["weight"] for subm in params["subm"]]
    mms, acc_mass = [], []

    for j, one_dossier in enumerate(dossiers):
        qs = [one["q"] for one in one_dossier["q_in"]]
        mm = [qs[h] * (main_wts[j] + sub_wts[h]) for h in range(len(sub_wts))]
        mass = sum(mm)
        mms.append(mm)
        acc_mass.append(round(mass))

    y_names = [name + " - " + str(mass) for name, mass in zip(subm_names, acc_mass)]
    f1 = figure(y_range=y_names, width=270, height=height, title="relations of general masses")
    f1.hbar(y=y_names, height=0.555, right=acc_mass, left=0, color=colors)
    output_file("tida_alls2.html")

    alls = [f"alls.{i}" for i in range(len(dossiers))]
    subm = [f"sub{i}" for i in range(len(dossiers))]
    mmsT = np.asarray(mms).T
    data = {"cols": alls}
    for i in range(len(dossiers)):
        data[f"sub{i}"] = mmsT[i, :]
    f2 = figure(y_range=alls, height=height, width=270, title="relations of columns masses")
    f2.hbar_stack(subm, y="cols", height=0.555, color=colors, source=data)

    qssT = np.asarray(qss).T
    data = {"cols": alls}
    for i in range(len(dossiers)):
        data[f"sub{i}"] = qssT[i, :]
    f3 = figure(y_range=alls, height=height, width=245, title="ratios in columns")
    f3.hbar_stack(subm, y="cols", height=0.555, color=colors, source=data)

    grid = gridplot([[f3, f2, f1]])
    show(grid)

    if show_figures2:
        def read(params, i):
            fin = params["path"] + params["subm"][i]["name"] + ".csv"
            target_name_back = {"target": params["target"], "pred": params["target"]}
            return pd.read_csv(fin).rename(columns=target_name_back)

        dfs = [read(params, i) for i in range(len(params["subm"]))] + [df_cross]
        _height = 358 if len(params["subm"]) == 11 else 254
        f = figure(width=785, height=_height)
        f.title.text = "Click on legend entries to mute the corresponding lines"
        b, e = 21000, 21121
        line_x = [dfs[i][b:e]["id"] for i in range(len(dfs))]
        line_y = [dfs[i][b:e]["diagnosed_diabetes"] for i in range(len(dfs))]
        color = colors + [color_cross]
        alpha = [0.8 for _ in range(len(dfs) - 1)] + [0.95]
        lws = [1.0 for _ in range(len(dfs) - 1)] + [1.00]
        legend = subm_names + ["cross"]
        for i in range(len(legend)):
            f.line(
                line_x[i],
                line_y[i],
                line_width=lws[i],
                color=color[i],
                alpha=alpha[i],
                muted_color="white",
                legend_label=legend[i],
            )
        f.legend.location = "top_left"
        f.legend.click_policy = "mute"
        show(f)



def matrix_vs(path, fs_names):
    def load_all():
        frames = []
        for name in fs_names:
            df = pd.read_csv(path + name + ".csv")
            frames.append(df.rename(columns={"diagnosed_diabetes": name}))
        merged = frames[0]
        for nxt in frames[1:]:
            merged = pd.merge(merged, nxt, on="id")
        return merged

    def pair_labels():
        pairs = []
        for i in range(len(fs_names) - 1):
            for j in range(i + 1, len(fs_names)):
                pairs.append(fs_names[i] + "_vs_" + fs_names[j])
        return pairs

    def with_distances(df, labels):
        for vs in labels:
            a, b = vs.split("_vs_")
            df[vs] = (df[a] - df[b]).abs()
        return df

    def distance_row(name, labels, df):
        distances = []
        for other in fs_names:
            vs = name + "_vs_" + other
            if vs not in labels:
                distances.append(0)
            else:
                distances.append(round(df[vs].sum()))
        return distances

    dfs = load_all()
    labels = pair_labels()
    dfs = with_distances(dfs, labels)
    m1 = pd.DataFrame({"subm": fs_names})
    m2 = pd.DataFrame({name: distance_row(name, labels, dfs) for name in fs_names})
    matrix = pd.concat([m1, m2], axis=1)
    return matrix



def seaborn_Show(params, file_name_cross=""):
    import matplotlib.pyplot as plt, seaborn as sns
    import warnings; warnings.filterwarnings("ignore")
    plt.figure(figsize=(9, 3))
    for subm in params["subm"]:
        pred = pd.read_csv(params["path"] + subm["name"] + ".csv")[params["id_target"][1]]
        sns.kdeplot(pred, label=subm["name"], linewidth=0.5)
    if file_name_cross != "":
        pred = pd.read_csv(file_name_cross)[params["id_target"][1]]
        sns.kdeplot(pred, label="blend", linewidth=1, linestyle="dashed")
    plt.title("KDE")
    plt.xlabel("target")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



def display_distances(params):
    files = [subm["name"] for subm in params["subm"]]
    distances = matrix_vs(params["path"], files)
    display(distances)



def arr_colors(color):
    sg = ["silver", "gainsboro"]
    palette = {
        "red": ["red", "crimson", "firebrick"],
        "r": ["red", "crimson", "firebrick"],
        "Red": ["red", "tomato", "crimson"],
        "R": ["red", "tomato", "crimson"],
        "Green": ["forestgreen", "limegreen", "darkgreen"],
        "G": ["forestgreen", "limegreen", "darkgreen"],
        "Blue": ["blue", "royalblue", "mediumblue"],
        "B": ["blue", "royalblue", "mediumblue"],
        "RGB": ["mediumblue", "darkgreen", "crimson"],
        "S": ["mediumblue", "darkgreen", "crimson"],
    }
    return palette.get(color, ["black", "dimgray", "gray"]) + sg



def convert(schema):
    colors = arr_colors(schema[2])
    dicts = [
        {"name": schema[0][i], "weight": schema[1][i], "color": colors[i]}
        for i in range(len(schema[0]))
    ]
    return {"subm": dicts}



def h_blend(
        params, _update={},
        cross="silver",
        details=False,
        fig1=False, fig2=False, wf2=555,
        dtls=False, dist=False, subm=""):

    if isinstance(params, list):
        params = convert(params)

    if "path" in _update:
        params.update(_update)

    color_cross = cross
    cfg = copy.deepcopy(params)

    if details:
        dist = True
        show_details, show_figures1, show_figures2 = True, True, True
    else:
        show_details, show_figures1, show_figures2 = dtls, fig1, fig2

    type_sort = cfg["type_sort"][0]
    cfg["asc"] = cfg["type_sort"][1]
    cfg["desc"] = cfg["type_sort"][2]
    cfg["id"] = cfg["id_target"][0]
    cfg["target"] = cfg["id_target"][1]

    def read_one(i):
        name = cfg["subm"][i]["name"]
        fin = cfg["path"] + name + ".csv"
        df = pd.read_csv(fin)
        return df.rename(columns={"target": name, "pred": name, cfg["target"]: name})

    def merge_all(frames):
        merged = frames[0]
        for nxt in frames[1:]:
            merged = pd.merge(merged, nxt, on=[cfg["id"]])
        return merged

    def ordering(row, direction, cols):
        if type_sort != "asc/desc":
            import random
            names = list(cols)
            random.shuffle(names)
            return names
        reverse = True if direction == "desc" else False
        pairs = {c: row[c] for c in cols}.items()
        return [p[0] for p in sorted(pairs, key=lambda k: k[1], reverse=reverse)]

    def score_row(row, cols, w_main, w_sub):
        order = row["alls"]
        total = 0
        for idx, col in enumerate(cols):
            rank = order.index(col)
            total += row[col] * (w_main[idx] + w_sub[rank])
        return total

    def compute(direction, show_details):
        df = merge_all([read_one(i) for i in range(len(cfg["subm"]))])
        cols = [c for c in df.columns if c != cfg["id"]]

        if "subwts2" in cfg or "subm2" in cfg:
            df["mx-m"] = df[cols].apply(lambda r: abs(r.max() - r.min()), axis=1)

        df["alls"] = df.apply(lambda r: ordering(r, direction, cols), axis=1)

        if "subwts2" not in cfg:
            main = [s["weight"] for s in cfg["subm"]]
            sub_wts = list(cfg["subwts"])
            df[cfg["target"]] = df.apply(lambda r: score_row(r, cols, main, sub_wts), axis=1)
        else:
            wts = [
                ([s["weight"] for s in cfg["subm"]], list(cfg["subwts"])),
                ([s["weight"] for s in cfg["subm2"]], list(cfg["subwts2"])),
                ([s["weight"] for s in cfg["subm3"]], list(cfg["subwts3"])),
            ]

            def choose_score(r):
                delta = r["mx-m"]
                if 0.00000 < delta <= 0.00100:
                    main, sub_wts = wts[1]
                elif 0.00100 < delta <= 0.00200:
                    main, sub_wts = wts[0]
                else:
                    main, sub_wts = wts[2]
                return score_row(r, cols, main, sub_wts)

            df[cfg["target"]] = df.apply(choose_score, axis=1)

        view = df.rename(columns={cfg["target"]: "ensemble"})
        view.insert(loc=1, column=" _ ", value=["   "] * len(view))
        view[" _ "] = view[" _ "].astype(str)
        pd.set_option("display.max_rows", 100)
        pd.set_option("display.float_format", "{:.5f}".format)
        if "subwts2" in cfg:
            vcols = [cfg["id"], " _ "] + cols + [" _ ", "mx-m", " _ ", "alls", " _ ", "ensemble"]
        else:
            vcols = [cfg["id"], " _ "] + cols + [" _ ", "alls", " _ ", "ensemble"]
        view = view[vcols]
        if show_details and direction == "desc":
            display(view.head(5))
        pd.set_option("display.float_format", "{:.5f}".format)
        view = view.rename(columns={"ensemble": cfg["target"]})
        if direction == "desc":
            view.to_csv(f"tida_{direction}.csv", index=False)
        return view[[cfg["id"], cfg["target"]]]

    df_desc = compute("desc", show_details)
    df_asc = compute("asc", show_details)
    df_asc[cfg["target"]] = cfg["desc"] * df_desc[cfg["target"]] + df_asc[cfg["target"]] * cfg["asc"]

    bokeh_show(cfg, df_asc, show_figures1, show_figures2, wf2, color_cross)

    if subm != "":
        df_asc.to_csv(subm, index=False)

    if dist:
        added = False
        if subm != "" and "/kaggle/working/" in params["path"]:
            params["subm"].append({"name": subm.replace(".csv", "").replace(params["path"], "")})
            added = True
        display_distances(params)
        cross_fin = subm if not added else ""
        seaborn_Show(params, file_name_cross=cross_fin)

    return df_asc



params = {
    "path": f"/kaggle/input/24-december-2025-ps-s5e12/0.7073",
    "id_target": ["id", "diagnosed_diabetes"],
    "type_sort": ["asc/desc", 0.30, 0.70],
    "different": [0, 1, 2],
    "subwts": [w / 100 for w in [-8, -5, 13]],
    "subm": [
        {"name": "a", "weight": 0.315, "color": "royalblue"},
        {"name": "c", "weight": 0.33, "color": "orange"},
        {"name": "d", "weight": 0.355, "color": "green"},
    ],
    "subwts2": [w / 100 for w in [-5, 13, -8]],
    "subm2": [
        {"name": "a", "weight": 0.315, "color": "royalblue"},
        {"name": "c", "weight": 0.33, "color": "orange"},
        {"name": "d", "weight": 0.355, "color": "green"},
    ],
    "subwts3": [w / 100 for w in [13, -8, -5]],
    "subm3": [
        {"name": "a", "weight": 0.315, "color": "royalblue"},
        {"name": "c", "weight": 0.33, "color": "orange"},
        {"name": "d", "weight": 0.355, "color": "green  "},
    ],
}

df = h_blend(params, details=True, subm="h-blend.csv")



import seaborn as sns
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

mx_m = pd.read_csv("tida_desc.csv")["mx-m"]

sns.set()
fig, ax = plt.subplots(figsize=(9, 3))
ax.hist(mx_m, bins=250)
ax.set_facecolor("whitesmoke")
fig.suptitle("mx-m", y=0.85, fontsize=11, color="black")



for name in ["h-blend", "tida_desc"]:
    os.remove(f"{name}.csv")



df.to_csv("submission.csv", index=False)
df





