!pip install goatools
!pip install pyvis


!python -V


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import textwrap

from collections import deque, defaultdict
from IPython.display import IFrame, display
from pyvis.network import Network
from pathlib import Path
from textwrap import fill
from goatools.obo_parser import GODag


CAFA6_DIR = "/kaggle/input/cafa-6-protein-function-prediction"
obo_path = Path(CAFA6_DIR, "Train/go-basic.obo")
go_dag = GODag(str(obo_path), optional_attrs={"relationship"})


type(go_dag)


list(go_dag.keys())[:5]


term = go_dag["GO:0008150"]
term


unique_terms = list({term.id: term for term in go_dag.values()}.values())
non_obsolete_terms = [t for t in unique_terms if not getattr(t, "is_obsolete", False)]

roots_all = sorted(
    [t for t in non_obsolete_terms if len(getattr(t, "parents", [])) == 0],
    key=lambda term: term.id,
)

roots_df = pd.DataFrame(
    [{"go_id": t.id, "namespace": t.namespace} for t in roots_all]
)
roots_df


namespace_map = {
    "molecular_function": "MF",
    "cellular_component": "CC",
    "biological_process": "BP",
}
domain_order = ["BP", "MF", "CC"]
colors = ["#4C78A8", "#F58518", "#54A24B"]

terms = {term.id: term for term in go_dag.values()}.values()
domains = [
    namespace_map.get(term.namespace)
    for term in terms
    if not getattr(term, "is_obsolete", False)
]
counts = (
    pd.Series(domains, name="domain")
    .value_counts().reindex(domain_order)
)

ax = counts.plot(kind="bar", color=colors, figsize=(6, 4), width=0.65)
ax.set_xlabel("")
ax.set_ylabel("Number of non-obsolete GO terms")
ax.set_title("GO term counts by domain (from go-basic.obo)")
ax.tick_params(axis="x", labelrotation=0)

y_max = counts.max()
ax.set_ylim(0, int(y_max * 1.12))

for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02 * y_max,
        f"{int(height):,}",
        ha="center",
        va="bottom",
        fontsize=11,
    )


relationship_counts = {}
is_a_count = 0

for term in non_obsolete_terms:
    is_a_count += len(getattr(term, "parents", []))
    relationships = getattr(term, "relationship", None) or {}
    for rel_type, rel_parents in relationships.items():
        relationship_counts[rel_type] = relationship_counts.get(rel_type, 0)
        relationship_counts[rel_type] += len(rel_parents)

counts_df = (
    pd.DataFrame(
        [{"type": "is_a", "n_edges": is_a_count}]
        + [{"type": rel_type, "n_edges": count}
           for rel_type, count in relationship_counts.items()]
    )
    .sort_values("n_edges", ascending=False)
    .reset_index(drop=True)
)

counts_df


def unique_non_obsolete_terms(go_dag):
    terms = list({term.id: term for term in go_dag.values()}.values())
    return [t for t in terms if not getattr(t, "is_obsolete", False)]

def build_domain_table(go_dag, domain, terms_subset=None):
    domain_to_namespace = {v: k for k, v in namespace_map.items()}
    
    terms = unique_non_obsolete_terms(go_dag)
    target_ns = domain_to_namespace[domain]
    terms = [t for t in terms if t.namespace == target_ns]
    if terms_subset is not None:
        terms = [t for t in terms if t.id in terms_subset]

    term_ids = {t.id for t in terms}

    in_all = {term_id: 0 for term_id in term_ids}
    out_all = {term_id: 0 for term_id in term_ids}

    for term in terms:
        children = [c.id for c in getattr(term, "children", []) if c.id in term_ids]
        out_all[term.id] += len(children)
        for child_id in children:
            in_all[child_id] += 1

        relationships = getattr(term, "relationship", None)
        for rel_type, rel_targets in relationships.items():
            targets = [x.id for x in rel_targets if x.id in term_ids]
            out_all[term.id] += len(targets)
            for target_id in targets:
                in_all[target_id] += 1

    domain_df = pd.DataFrame({
        "go_id": [t.id for t in terms],
        "name": [t.name for t in terms],
        "depth_is_a": [int(getattr(t, "depth", -1)) for t in terms],
        "n_parents_is_a": [len(getattr(t, "parents", [])) for t in terms],
        "n_children_is_a": [len(getattr(t, "children", [])) for t in terms],
        "n_in_all": [in_all[t.id] for t in terms],
        "n_out_all": [out_all[t.id] for t in terms],
    })
    domain_df["n_edges_all"] = domain_df["n_in_all"] + domain_df["n_out_all"]
    return domain_df

def summarize_domain_table(domain_df):
    columns = [
        "n_in_all",
        "n_out_all",
        "n_edges_all",
        "n_parents_is_a",
        "n_children_is_a",
        "depth_is_a",
    ]
    return domain_df[columns].describe(percentiles=[0.5]).T

def plot_edge_hist_log10(domain_dfs, bins=60):
    fig, axes = plt.subplots(
        nrows=len(domain_dfs),
        ncols=1,
        figsize=(10, 4 * len(domain_dfs)),
        sharex=True
    )

    ticks = [0, np.log10(2), np.log10(5), 1, 2]
    labels = ["0", "1", "4", "9", "99"]

    for ax, (name, domain_df) in zip(axes, domain_dfs.items()):
        values = domain_df["n_edges_all"].to_numpy(dtype=float)
        values_log = np.log10(values + 1.0)
        median_log = float(np.median(values_log))

        ax.hist(values_log, bins=bins,
                color="steelblue", edgecolor="black")
        ax.axvline(median_log, linestyle="--", color="red")

        ax.set_ylabel("Count")
        ax.set_title(f"{name} ontology")

    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel(
        "Number of edges per term (log10 transform)"
    )

    plt.tight_layout()
    return axes

def build_k_level_graph(go_dag, root_id, max_depth=3):
    graph = nx.DiGraph()
    level_by_id = {root_id: 1}
    queue = deque([root_id])
    graph.add_node(root_id, level=1)

    while queue:
        node_id = queue.popleft()
        node_level = level_by_id[node_id]
        if node_level >= max_depth:
            continue

        node = go_dag[node_id]
        children_by_rel = defaultdict(set)

        for child in getattr(node, "children", []) or []:
            if not getattr(child, "is_obsolete", False):
                children_by_rel["is_a"].add(child)

        rel_rev = getattr(node, "relationship_rev", None) or {}
        for rel_type, rel_terms in rel_rev.items():
            for child in rel_terms:
                if not getattr(child, "is_obsolete", False):
                    children_by_rel[rel_type].add(child)

        next_level = node_level + 1
        for rel_type, children in children_by_rel.items():
            for child in sorted(children, key=lambda t: t.id):
                child_id = child.id
                graph.add_edge(node_id, child_id, rel=rel_type)
                prev_level = level_by_id.get(child_id)
                if prev_level is None or next_level < prev_level:
                    level_by_id[child_id] = next_level
                    graph.add_node(child_id, level=next_level)
                    queue.append(child_id)

    node_set = set(graph.nodes())
    for node_id in list(node_set):
        node = go_dag[node_id]
        rel_rev = getattr(node, "relationship_rev", None) or {}
        for rel_type, rel_terms in rel_rev.items():
            for child in rel_terms:
                if getattr(child, "is_obsolete", False):
                    continue
                if child.id in node_set:
                    graph.add_edge(node_id, child.id, rel=rel_type)

    return graph


def pyvis_render(go_dag, graph, wrap_label=False, title="GO DAG"):
    net = Network(
        height="750px", width="100%", directed=True, notebook=True,
        cdn_resources="in_line",
    )
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 320,
          "nodeSpacing": %d,
          "treeSpacing": 120,
          "blockShifting": true,
          "edgeMinimization": true,
          "parentCentralization": true
        }
      },
      "physics": { "enabled": false },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      },
      "nodes": {
        "shape": "box",
        "font": { "size": 11, "multi": %s, "face": "arial" },
        "margin": { "top": 2, "right": 4, "bottom": 2, "left": 4 }
      },
      "edges": {
        "smooth": false,
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }
      }
    }
    """ % (
        60 if wrap_label else 30,
        "true" if wrap_label else "false"
    ))

    rel_color = {
        "is_a": "#1C588C",
        "part_of": "black",
        "has_part": "gray",
        "regulates": "#BF9039",
        "negatively_regulates": "#BF5841",
        "positively_regulates": "#2D735F",
    }

    for node_id, data in graph.nodes(data=True):
        term = go_dag[node_id]
        level = int(data.get("level", 1))
        full_label = f"{node_id}: {term.name}"
        label = ("\n".join(textwrap.wrap(full_label, width=28))
         if wrap_label else
         (full_label if len(full_label) <= 80 else full_label[:77] + "..."))
        net.add_node(
            node_id,
            label=label,
            title=f"{node_id}<br>{term.name}<br>level: {level}",
            shape="box",
            level=level,
        )

    used_rels = set()
    for u, v, data in graph.edges(data=True):
        rel = data.get("rel", "is_a")
        used_rels.add(rel)
        net.add_edge(u, v, title=rel, color=rel_color.get(rel, "#999999"))

    legend = " &nbsp; ".join(
        "<span style='display:inline-block;width:12px;height:12px;"
        f"background:{rel_color.get(r, '#999999')};margin-right:6px;'></span>{r}"
        for r in sorted(used_rels)
    )

    html = net.generate_html()
    header = (
        f"<h3 style='margin:0 0 6px 0;'>{title}</h3>"
        f"<div style='margin:0 0 10px 0;font-size:14px;'>{legend}</div>"
    )
    
    html = html.replace("<body>", "<body>" + header, 1)
    
    path = f"pyvis_graph_{id(graph)}.html"
    with open(path, "w", encoding="utf-8") as file:
        file.write(html)
    
    return IFrame(path, width="100%", height=800)


bp_df = build_domain_table(go_dag, domain="BP")
summarize_domain_table(bp_df)


mf_df = build_domain_table(go_dag, domain="MF")
summarize_domain_table(mf_df)


cc_df = build_domain_table(go_dag, domain="CC")
summarize_domain_table(cc_df)


plot_edge_hist_log10({
    "Biological Process": bp_df,
    "Molecular Function": mf_df,
    "Cellular Component": cc_df
})


graph = build_k_level_graph(go_dag, root_id = "GO:0008150", max_depth=2)
title = "Top-level BP GO terms (edges by relationship type)"
HTML_obj = pyvis_render(go_dag, graph, title=title)
display(HTML_obj)


def load_go_df(obo_path, optional_attrs=None):
    optional_attrs = set(optional_attrs or set()) | {"relationship", "def"}
    go_dag = GODag(str(obo_path), optional_attrs=optional_attrs, load_obsolete=True)
    rows = []
    for go_id, term in go_dag.items():
        rows.append(
            {
                "GOID": go_id,
                "name": getattr(term, "name", None),
                "definition": getattr(term, "defn", None),
                "is_obsolete": bool(getattr(term, "is_obsolete", False)),
            }
        )
    return pd.DataFrame(rows)


def compare_go_obo(old_obo, new_obo):
    old_df = load_go_df(old_obo)
    new_df = load_go_df(new_obo)

    old_terms = set(old_df["GOID"])
    new_terms = set(new_df["GOID"])
    added_terms = new_terms - old_terms
    removed_terms = old_terms - new_terms

    obsolete_old = set(old_df.loc[old_df["is_obsolete"], "GOID"])
    obsolete_new = set(new_df.loc[new_df["is_obsolete"], "GOID"])
    new_obsoletes = obsolete_new - obsolete_old
    added_as_obsolete = added_terms & obsolete_new

    df = old_df.merge(
        new_df, on="GOID", how="outer", suffixes=("_old", "_new"), validate="one_to_one"
    )

    in_overlap = df["name_old"].notna() & df["name_new"].notna()
    def_overlap = df["definition_old"].notna() & df["definition_new"].notna()
    newly_added_defs = in_overlap & df["definition_old"].isna() & df["definition_new"].notna()

    df["name_changed"] = in_overlap & df["name_old"].ne(df["name_new"])
    df["definition_changed"] = def_overlap & df["definition_old"].ne(
        df["definition_new"]
    )

    old_len = df["definition_old"].fillna("").astype(str).str.len()
    new_len = df["definition_new"].fillna("").astype(str).str.len()
    changed_def = df["definition_changed"]

    df["definition_delta"] = np.select(
        [changed_def & (new_len > old_len),
         changed_def & (new_len < old_len),
         changed_def & (new_len == old_len)],
        ["Extended", "Simplified", "Same length"],
        default=pd.NA,
    )

    df["term_status"] = np.select(
        [df["GOID"].isin(added_terms),
        df["GOID"].isin(removed_terms),
        df["GOID"].isin(new_obsoletes)],
        ["Added term", "Removed term", "Newly obsolete"],
        default="Present in both",
    )

    metrics = pd.Series(
        {
            "Terms (old)": len(old_terms),
            "Terms (new)": len(new_terms),
            "Obsolete terms (old)": int(old_df["is_obsolete"].sum()),
            "Obsolete terms (new)": int(new_df["is_obsolete"].sum()),
            "Added terms": len(added_terms),
            "Added terms (obsolete)": len(added_as_obsolete),
            "Removed terms": len(removed_terms),
            "Newly obsolete terms": len(new_obsoletes),
            "Newly added definitions": int(newly_added_defs.sum()),
            "Name changed (overlap)": int(df["name_changed"].sum()),
            "Definition changed (overlap)": int(df["definition_changed"].sum()),
            "Definition simplified (overlap)": int(
                (df["definition_delta"] == "Simplified").sum()
            ),
            "Definition extended (overlap)": int(
                (df["definition_delta"] == "Extended").sum()
            ),
        },
        name="N",
    )

    summary_fields = metrics.rename_axis("metric").reset_index()

    return {"summary_fields": summary_fields, "df": df}


def plot_term_status(df, title):
    counts = (
        df["term_status"].value_counts()
        .rename_axis("term_status")
        .reset_index(name="N")
    )
    ax = counts.plot(
        kind="bar", x="term_status", y="N", figsize=(5, 3),
        width=0.65, color="steelblue", legend=False
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of terms")
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=0)
    
    y_max = counts["N"].max()
    ax.set_ylim(0, int(y_max * 1.12))
    
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02 * y_max,
                f"{int(height):,}", ha="center", va="bottom", fontsize=11)
    return ax


CAFA6_DIR = "/kaggle/input/cafa-6-protein-function-prediction"
CAFA5_DIR = "/kaggle/input/cafa-5-protein-function-prediction"
obo_current = Path(CAFA6_DIR, "Train/go-basic.obo")
obo_cafa5 = Path(CAFA5_DIR, "Train/go-basic.obo")

res = compare_go_obo(obo_cafa5, obo_current)
display(res["summary_fields"])


res["df"].loc[
    res["df"]["definition_changed"],
    ["GOID", "definition_old", "definition_new"]
]


train_path = Path(CAFA6_DIR, "Train/train_terms.tsv")
train_set = pd.read_csv(
    train_path, sep="\t", dtype={
        "EntryID": "string",
        "term": "string",
        "aspect": "string"}
    )

aspect_map = {"P": "BP", "F": "MF", "C": "CC", "BPO": "BP", "MFO": "MF", "CCO": "CC"}
train_set["domain"] = train_set["aspect"].map(aspect_map).fillna(train_set["aspect"])

print(f"The head of the dataset with {len(train_set):,} rows")
display(train_set.head())


n_proteins = train_set["EntryID"].nunique()
n_terms = train_set["term"].nunique()
print(f"Unique proteins: {n_proteins:,}")
print(f"Unique GO terms: {n_terms:,}")

per_protein = train_set.groupby("EntryID", sort=False)["term"].nunique()
n_le_10 = int((per_protein <= 10).sum())
print("Number of GO terms per protein:")
print(f"range: {int(per_protein.min())} to {int(per_protein.max())}")
print(f"mean: {per_protein.mean():.3f}")
print(f"median: {per_protein.median():.0f}")
print(f"Number of proteins with 10 or lesser GO terms: {n_le_10:,} "
      f"({n_le_10 / n_proteins:.2%} of proteins)")

per_term = train_set.groupby("term", sort=False)["EntryID"].nunique()
n_singletons = int((per_term == 1).sum())
print(f"Number of GO terms assigned to only one protein: {n_singletons:,} "
      f"({n_singletons / n_terms:.2%} of GO terms)")

by_aspect = train_set.groupby("domain").agg(
    n_Proteins=("EntryID", "nunique"),
    n_GOterms=("term", "nunique"),
).reset_index()
print("\nNumber of unique proteins and GO terms by ontology:")
display(by_aspect)


terms_per_prot = (
    train_set.groupby(["EntryID", "domain"], sort=False)["term"]
    .nunique()
    .reset_index(name="n_terms")
)

median_by_domain = terms_per_prot.groupby("domain")["n_terms"].median()

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, domain in zip(axes, namespace_map.values()):
    data = terms_per_prot.loc[
        terms_per_prot["domain"] == domain, "n_terms"
    ]
    counts = data.value_counts().sort_index()
    ax.bar(counts.index, counts.values)
    ax.set_yscale("log")
    ax.set_title(domain)
    ax.set_xlabel("Number of GO terms per protein")

    median_val = int(median_by_domain.loc[domain])
    ax.axvline(median_val, color="crimson", lw=1)

    xticks = list(ax.get_xticks())
    if median_val not in xticks:
        xticks.append(median_val)
        xticks = sorted(xticks)
        ax.set_xticks(xticks)

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if tick == median_val:
            label.set_color("crimson")

axes[0].set_ylabel("Number of proteins")
plt.tight_layout()
plt.show()


def term_table(train_df, domain, k=10, rare=False):
    subset = train_df.loc[train_df["domain"].eq(domain), ["term", "EntryID"]]
    if subset.empty:
        return pd.DataFrame(
            {"GO_id": pd.Series([], dtype="string"), "n_proteins": pd.Series([], dtype="int64")}
        )
    counts = subset.groupby("term", sort=False)["EntryID"].size()
    counts = counts.sort_values(ascending=rare).head(k)
    return counts.rename("n_proteins").reset_index().rename(columns={"term": "GO_id"})

def annotate_go_table(go_dag, df, wrap_width=35):
    go_ids = df["GO_id"].tolist()
    df = df.copy()
    df["GO_term"] = [
        fill(getattr(go_dag.get(go_id), "name", ""), width=wrap_width) for go_id in go_ids
    ]
    df["level_in_DAG"] = [
        getattr(go_dag.get(go_id), "level", np.nan)
        for go_id in go_ids
    ]
    df["n_ANCESTORS"] = [
        len(getattr(go_dag.get(go_id), "get_all_parents", lambda: set())())
        for go_id in go_ids
    ]
    df["n_PARENTS"] = [
        len(getattr(go_dag.get(go_id), "parents", {}))
        for go_id in go_ids
    ]
    df["n_CHILDREN"] = [
        len(getattr(go_dag.get(go_id), "children", []))
        for go_id in go_ids
    ]
    df["n_OFFSPRINGS"] = [
        len(getattr(go_dag.get(go_id), "get_all_children", lambda: set())())
        for go_id in go_ids
    ]
    return df


bp_common = term_table(train_set, domain="BP", k=10, rare=False)
bp_common = annotate_go_table(go_dag, bp_common, wrap_width=35)
display(bp_common)


bp_rare = term_table(train_set, "BP", k=10, rare=True)
bp_rare = annotate_go_table(go_dag, bp_rare, wrap_width=35)
display(bp_rare)


term_counts = train_set.groupby("term", sort=False)["EntryID"].nunique()
term_meta = term_counts.rename("n_proteins").reset_index().rename(
    columns={"term": "GO_id"}
)
term_meta["GO_term"] = term_meta["GO_id"].map(
    lambda go_id: go_dag[go_id].name if go_id in go_dag else pd.NA
)
term_meta["depth"] = term_meta["GO_id"].map(
    lambda go_id: go_dag[go_id].depth if go_id in go_dag else pd.NA
)
term_meta["domain"] = term_meta["GO_id"].map(
    lambda go_id: namespace_map.get(go_dag[go_id].namespace, pd.NA)
    if go_id in go_dag else pd.NA
)
term_meta


fig, ax = plt.subplots(figsize=(14, 5))

ax.hist(
    term_meta["n_proteins"],
    bins=50,
    log=True
)
ax.set_xlabel("Number of proteins assigned a GO term")
ax.set_ylabel("Number of GO terms")
ax.set_title("Distribution of proteins per GO term")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, domain in zip(axes, namespace_map.values()):
    depth_counts = (
        term_meta.loc[term_meta["domain"] == domain]
        .dropna(subset=["depth"])
        .groupby("depth", sort=True)
        .size()
    )
    ax.bar(depth_counts.index, depth_counts.values)
    ax.set_title(domain)
    ax.set_xlabel("GO term depth")

axes[0].set_ylabel("Number of GO terms")

plt.tight_layout()
plt.show()


from functools import lru_cache

@lru_cache(maxsize=None)
def all_parents(go_id: str) -> frozenset[str]:
    if go_id not in go_dag:
        return frozenset()
    return frozenset(go_dag[go_id].get_all_parents())


def protein_go_dependency_stats(annotations: pd.DataFrame) -> pd.DataFrame:
    protein_to_terms = (
        annotations.groupby("EntryID", sort=False)["term"]
        .apply(lambda s: set(s.dropna().unique()))
    )

    rows = []
    for entry_id, term_set in protein_to_terms.items():
        valid_terms = {go_id for go_id in term_set if go_id in go_dag}
        if not valid_terms:
            rows.append({
                "EntryID": entry_id,
                "n_terms": 0,
                "n_dependent_terms": 0,
                "share_dependent_terms": 0.0,
                "has_any_dependency": False,
            })
            continue

        dependent_terms = set()
        for go_id in valid_terms:
            if all_parents(go_id) & valid_terms:
                dependent_terms.add(go_id)

        n_terms = len(valid_terms)
        n_dependent = len(dependent_terms)

        rows.append({
            "EntryID": entry_id,
            "n_terms": n_terms,
            "n_dependent_terms": n_dependent,
            "share_dependent_terms": n_dependent / n_terms,
            "has_any_dependency": n_dependent > 0,
        })

    return pd.DataFrame(rows)


protein_stats = protein_go_dependency_stats(train_set)
protein_stats.sort_values(
    ["n_dependent_terms", "share_dependent_terms"],
    ascending=False
)


summary = pd.Series({
    "n_proteins": len(protein_stats),
    "share_with_any_dependency":
        protein_stats["has_any_dependency"].mean(),
    "median_terms_per_protein":
        protein_stats["n_terms"].median(),
    "median_share_dependent_terms":
        protein_stats["share_dependent_terms"].median(),
})
summary


GO_id = "GO:0006357"

go_term = go_dag.get(GO_id)
go_term


print("\n\nProteins having this GO term in their annotation")
train_set.loc[train_set["term"] == GO_id]


def build_term_family_graph(go_dag, term_id):
    graph = nx.DiGraph()
    term = go_dag[term_id]

    parents_by_rel = defaultdict(set)
    for parent in getattr(term, "parents", []) or []:
        if not getattr(parent, "is_obsolete", False):
            parents_by_rel["is_a"].add(parent)

    relationships = getattr(term, "relationship", None) or {}
    for rel_type, rel_terms in relationships.items():
        for parent in rel_terms:
            if not getattr(parent, "is_obsolete", False):
                parents_by_rel[rel_type].add(parent)

    children_by_rel = defaultdict(set)
    for child in getattr(term, "children", []) or []:
        if not getattr(child, "is_obsolete", False):
            children_by_rel["is_a"].add(child)

    rel_rev = getattr(term, "relationship_rev", None) or {}
    for rel_type, rel_terms in rel_rev.items():
        for child in rel_terms:
            if not getattr(child, "is_obsolete", False):
                children_by_rel[rel_type].add(child)

    graph.add_node(term_id, level=2)

    for rel_type, parents in parents_by_rel.items():
        for parent in sorted(parents, key=lambda t: t.id):
            graph.add_node(parent.id, level=1)
            graph.add_edge(parent.id, term_id, rel=rel_type)

    for rel_type, children in children_by_rel.items():
        for child in sorted(children, key=lambda t: t.id):
            graph.add_node(child.id, level=3)
            graph.add_edge(term_id, child.id, rel=rel_type)

    node_set = set(graph.nodes())
    for node_id in list(node_set):
        node = go_dag[node_id]
        rel_rev = getattr(node, "relationship_rev", None) or {}
        for rel_type, rel_terms in rel_rev.items():
            for child in rel_terms:
                if getattr(child, "is_obsolete", False):
                    continue
                if child.id in node_set:
                    graph.add_edge(node_id, child.id, rel=rel_type)

    return graph


graph = build_term_family_graph(go_dag, GO_id)
title = f"GO term family: {GO_id} ({go_dag[GO_id].name})"

HTML_obj = pyvis_render(go_dag, graph, wrap_label=True, title=title)
display(HTML_obj)

