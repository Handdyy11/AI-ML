"""SENTRIX dashboard. Run from the project root:  streamlit run dashboard.py"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config


# What the autonomous defense does for each predicted family
ACTIONS = {
    "Benign": "Allow the traffic",
    "PortScan": "Raise an alert and log the source IP",
    "BruteForce": "Block the source IP",
    "DoS": "Rate-limit or block the source IP",
}

# Features that only take a few fixed values -> shown as a dropdown
CHOICES = {
    "Protocol": {6: "6 (TCP)", 17: "17 (UDP)"},
    "PSH Flag Count": {0: "0 (no PSH flag)", 1: "1 (PSH flag set)"},
}

def read_json(name):
    path = config.RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_family_counts():
    train = pd.read_parquet(config.DATA_PROCESSED / "train.parquet", columns=["family"])
    test = pd.read_parquet(config.DATA_PROCESSED / "test.parquet", columns=["family"])
    return pd.concat([train, test])["family"].value_counts()


@st.cache_data
def load_test():
    return pd.read_parquet(config.DATA_PROCESSED / "test.parquet")



@st.cache_resource
def load_models():
    knn = joblib.load(config.DATA_PROCESSED / "knn_model.joblib")

    tree_path = config.DATA_PROCESSED / "tree_model.joblib"
    tree = joblib.load(tree_path) if tree_path.exists() else None

    forest_path = config.DATA_PROCESSED / "forest_model.joblib"
    forest = joblib.load(forest_path) if forest_path.exists() else None

    scaler = joblib.load(config.DATA_PROCESSED / "scaler.joblib")

    return knn, tree, forest, scaler



def plot_confusion(cm, labels, title):
    cm = np.array(cm)
    shares = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if shares[i, j] > 0.5 else "black"
            ax.text(j, i, f"{shares[i, j]:.1%}\n({cm[i, j]:,})",
                    ha="center", va="center", fontsize=7, color=colour)
    fig.tight_layout()
    return fig


def per_family_table(m):
    return pd.DataFrame(m["per_family"]).T.rename(columns={"f1-score": "f1"})


def run_model(bundle, scaler, flow, cols):
    X = scaler.transform(flow[cols]).astype("float32")
    X = bundle["selector"].transform(X)
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    k = int(np.argmax(proba))
    return model.classes_[k], float(proba[k])


# ---------------------------------------------------------------- page 1
def page_data():
    st.header("1. Dataset and preprocessing")
    stats = read_json("preprocessing_stats.json")
    if stats is None:
        st.error("Run `python -m src.p2_preprocess.clean` first.")
        return

    st.subheader("CICIDS2017 files used")
    files = pd.DataFrame(
        [{"File": n, "Rows": v["rows"], "Columns": v["columns"]}
         for n, v in stats["files"].items()]
    )
    st.dataframe(files, hide_index=True, use_container_width=True)

    r1 = st.columns(3)
    r1[0].metric("Raw rows", f"{stats['rows_raw']:,}")
    r1[1].metric("Missing / inf rows removed", f"{stats['missing_removed']:,}")
    r1[2].metric("Duplicates removed", f"{stats['duplicates_removed']:,}")

    r2 = st.columns(3)
    r2[0].metric("Clean rows", f"{stats['rows_clean']:,}")
    r2[1].metric("Rows in our sample", f"{stats['rows_sample']:,}")
    r2[2].metric("Sample share", f"{stats['sample_fraction']:.1%}")
    st.markdown(
        f"**Why a sample?** The full data is too big and k-NN gets very slow on it, "
        f"so we kept {stats['sample_fraction']:.1%} of the rows. The sample is stratified, "
        f"so every attack family keeps the same share as in the full data."
    )
    st.markdown(
        "**How we preprocessed:** (1) Cleaned each file: fixed the column names, removed "
        "the duplicate column, and dropped rows with missing or infinite values and duplicate rows. "
        "(2) Grouped the attack labels into 4 families, split 80% train / 20% test with the same "
        "family share in both, and scaled the features using the training data only."
    )

    st.subheader("Rows per family")
    st.bar_chart(load_family_counts())

    with st.expander("Feature histograms"):
        pics = sorted((config.RESULTS / "figures").glob("hist_*.png"))
        grid = st.columns(2)
        for i, p in enumerate(pics):
            grid[i % 2].image(str(p))


# ---------------------------------------------------------------- page 2
def page_models():
    st.header("2. Classifiers")
    res = read_json("metrics_knn.json")
    if res is None:
        st.error("Run `python -m src.p3_supervised.knn_classifier` first.")
        return
    labels = res.get("labels", ["Benign", "BruteForce", "DoS", "PortScan"])

    st.subheader("k-NN: choosing k")
    if "tuning" in res:
        tune = pd.DataFrame(res["tuning"])
        st.dataframe(tune.sort_values("macro_f1", ascending=False),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Tuning table not saved yet. Re-run the k-NN script after Change 2.")
    st.success(f"Best k-NN: k = {res['knn']['k']}, weights = {res['knn']['weights']}")

    st.subheader("Test-set comparison")
    rows = {
        "Always say Benign": res["majority_baseline"],
        "Decision tree": res["decision_tree"],
        "k-NN": res["knn"],
        "Random Forest": res["random_forest"],
    }
    comp = pd.DataFrame(
        {n: {"Accuracy": m["accuracy"], "Macro F1": m["macro_f1"]} for n, m in rows.items()}
    ).T
    st.dataframe(comp.style.format("{:.4f}"))
    best = max(("Decision tree", "Random Forest", "k-NN"), key=lambda n: rows[n]["macro_f1"])
    st.success(f"Best model by macro F1: {best}")
    st.caption("Most traffic is Benign, so accuracy alone is misleading. Macro F1 gives every family equal weight.")

    st.subheader("Precision and recall per family")

    model_keys = [
        ("k-NN", "knn"),
        ("Decision tree", "decision_tree"),
        ("Random Forest", "random_forest"),
    ]
    for name, key in model_keys:
        with st.expander(name, expanded=(key == "random_forest")):
            st.dataframe(
                per_family_table(res[key]).style.format("{:.3f}")
            )

            if "confusion" in res[key]:
                st.pyplot(
                    plot_confusion(res[key]["confusion"], labels, name)
                )


# ---------------------------------------------------------------- page 3
# ---------------------------------------------------------------- page 3
@st.cache_data
def feature_guide(features):
    """Min/max and typical (5th-95th percentile) range per family for each feature."""
    test = load_test()
    rows = []
    for f in features:
        r = {"Feature": f, "Min": test[f].min(), "Max": test[f].max()}
        for fam, g in test.groupby("family")[f]:
            r[fam] = f"{g.quantile(0.05):,.2f} to {g.quantile(0.95):,.2f}"
        rows.append(r)
    return pd.DataFrame(rows).set_index("Feature")


@st.cache_data
def family_medians(features):
    test = load_test()
    return test.groupby("family")[list(features)].median().T


def flow_position(test, family):
    """Random flow from the chosen family (or from any family if 'Random')."""
    if family == "Random":
        family = np.random.choice(test["family"].unique())
    return int(np.random.choice(np.where(test["family"] == family)[0]))


def to_number(value):
    """Accept a number (from a dropdown) or text (from a text box)."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def page_predict():
    st.header("3. Try a network flow")
    res = read_json("metrics_knn.json")
    if res is None:
        st.error("Run the k-NN script first.")
        return
    chosen = res["features_used"]
    cols = json.loads((config.DATA_PROCESSED / "feature_columns.json").read_text())
    test = load_test()
    knn_b, tree_b, forest_b, scaler = load_models()
    guide = feature_guide(tuple(chosen))

    # ---- load a flow
    c1, c2 = st.columns([2, 1])
    family_choice = c1.selectbox(
        "Start from a flow of type", ["Random"] + sorted(test["family"].unique())
    )
    c2.write("")
    if "pos" not in st.session_state or c2.button("Load flow"):
        st.session_state.pos = flow_position(test, family_choice)

    pos = st.session_state.pos
    row = test.iloc[pos]
    src_ip = row.get("Source IP", "unknown")
    st.caption(f"Source IP of this flow: {src_ip}")

    # ---- feature guide
    with st.expander("Which features should I change? (typical values per family)"):
        st.markdown(
            "Each column shows the usual range (5th to 95th percentile) of that feature "
            "for one family. To make a flow look like an attack, move the values "
            "into that family's range."
        )
        st.dataframe(guide.style.format({"Min": "{:,.2f}", "Max": "{:,.2f}"}),
                     use_container_width=True)

    # ---- editable features (plain text boxes, no +/- buttons)
    st.markdown("Click a value, type a new one, then press **Predict**.")
    texts = {}
    
    grid = st.columns(4)

    for i, feat in enumerate(chosen):
        with grid[i % 4]:
            st.markdown(f"**{feat}**")

            if feat in CHOICES:
                options = list(CHOICES[feat])
                nearest = min(options, key=lambda o: abs(o - float(row[feat])))

                texts[feat] = st.selectbox(
                    feat,
                    options,
                    index=options.index(nearest),
                    format_func=CHOICES[feat].get,
                    key=f"{pos}_{feat}",
                    label_visibility="collapsed",
                )
                range_text = "Allowed: " + ", ".join(str(o) for o in options)

            else:
                texts[feat] = st.text_input(
                    feat,
                    value=f"{row[feat]:.4f}",
                    key=f"{pos}_{feat}",
                    label_visibility="collapsed",
                )
                range_text = (
                    f"Range: {guide.loc[feat, 'Min']:,.2f} to "
                    f"{guide.loc[feat, 'Max']:,.2f}"
                )

            st.markdown(
                f"<div style='height:42px; font-size:12px; "
                f"color:gray; overflow-wrap:anywhere;'>{range_text}</div>",
                unsafe_allow_html=True,
            )


    if st.button("Predict", type="primary"):
        values = {f: to_number(t) for f, t in texts.items()}
        bad = [f for f, v in values.items() if v is None]
        if bad:
            st.error("Enter plain numbers only. Check: " + ", ".join(bad))
            return
        outside = [f for f, v in values.items()
                   if f not in CHOICES
                   and not (guide.loc[f, "Min"] <= v <= guide.loc[f, "Max"])]
        if outside:
            st.warning("Outside the range seen in the data (prediction may be unreliable): "
                       + ", ".join(outside))

        flow = row[cols].astype("float64").to_frame().T
        for f, v in values.items():
            flow[f] = v

        res = read_json("metrics_knn.json")

        best_key = max(
            ["knn", "decision_tree", "random_forest"],
            key=lambda key: res[key]["macro_f1"]
        )

        best_models = {
            "knn": ("k-NN", knn_b),
            "decision_tree": ("Decision Tree", tree_b),
            "random_forest": ("Random Forest", forest_b),
        }

        best_name, best_bundle = best_models[best_key]

        if best_bundle is None:
            st.error(f"{best_name} model file is missing.")
            return

        models = [(best_name, best_bundle)]

        for col, (name, bundle) in zip(st.columns(len(models)), models):
            with col:
                st.subheader(name)
                pred, conf = run_model(bundle, scaler, flow, cols)
                st.metric("Predicted family", pred, f"{conf:.0%} confident", delta_color="off")
                st.markdown(f"**Autonomous action:** {ACTIONS.get(pred, 'Review manually')}")
                if pred != "Benign":
                    st.caption(f"Target: {src_ip}")
        

    with st.expander("Response rules"):
        st.dataframe(pd.DataFrame(list(ACTIONS.items()), columns=["Predicted family", "Action"]),
                     hide_index=True)

# ---------------------------------------------------------------- main
st.title("Autonomous Cyber-Defense & Intrusion Detection System")
page = st.sidebar.radio("Page", ["Dataset and preprocessing", "Classifiers", "Try a flow"])
{"Dataset and preprocessing": page_data,
 "Classifiers": page_models,
 "Try a flow": page_predict}[page]()        