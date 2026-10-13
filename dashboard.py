"""SENTRIX dashboard.  Run from the project root:  streamlit run dashboard.py"""
import json

import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config

# =============================================================== PAGE SETUP
st.set_page_config(page_title="SENTRIX", page_icon="🛡️", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, p, label, input, textarea, button, h1, h2, h3, h4, h5, h6,
.stMarkdown, [data-testid="stMetricValue"] {font-family: 'Inter', sans-serif;}
[data-testid="stIconMaterial"], .material-symbols-rounded {font-family: 'Material Symbols Rounded' !important;}
.block-container {padding-top: 1.2rem; max-width: 1400px;}
#MainMenu, footer {visibility: hidden;}

[data-testid="stSidebar"] {background: #0d1526; border-right: 1px solid rgba(255,255,255,.06);}

.hero {background: linear-gradient(135deg,#0ea5e9 0%,#6366f1 55%,#8b5cf6 100%);
       border-radius: 18px; padding: 26px 32px; margin-bottom: 22px;
       box-shadow: 0 10px 30px rgba(99,102,241,.25);}
.hero-t {font-size: 32px; font-weight: 800; color: white; letter-spacing: .1em;}
.hero-s {font-size: 15px; color: rgba(255,255,255,.9); margin-top: 4px;}

.kpi {background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
      border: 1px solid rgba(255,255,255,.08); border-radius: 14px;
      padding: 16px 18px; margin-bottom: 12px;}
.kl {font-size: 11px; text-transform: uppercase; letter-spacing: .1em; color: #9aa4b2; font-weight: 600;}
.kv {font-size: 27px; font-weight: 700; line-height: 1.3; color: #f1f5f9;}
.ks {font-size: 12px; color: #8b97a8;}

.section {font-size: 13px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
          color: #9aa4b2; margin: 8px 0 10px 2px;}

.verdict {border: 2px solid; border-radius: 16px; padding: 20px; text-align: center; margin-bottom: 14px;}
.vl {font-size: 11px; letter-spacing: .14em; color: #9aa4b2; font-weight: 600;}
.vv {font-size: 40px; font-weight: 800; line-height: 1.2;}
.vs {font-size: 13px; color: #cbd5e1;}
.pill {display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: 12px;
       font-weight: 700; letter-spacing: .1em; color: #0b1220; margin-top: 10px;}
.action {background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
         border-radius: 12px; padding: 12px 16px; margin-bottom: 12px;}
</style>
"""

# =============================================================== CONSTANTS
FAMILY_COLORS = {
    "Benign": "#22c55e",
    "PortScan": "#f59e0b",
    "BruteForce": "#f97316",
    "DoS": "#ef4444",
}

# What the autonomous defense does for each predicted family
ACTIONS = {
    "Benign": "Allow the traffic",
    "PortScan": "Raise an alert and log the source IP",
    "BruteForce": "Block the source IP",
    "DoS": "Rate-limit or block the source IP",
}
ACTION_TAG = {"Benign": "ALLOW", "PortScan": "ALERT", "BruteForce": "BLOCK", "DoS": "RATE-LIMIT"}

# Features that only take a few fixed values -> shown as a dropdown
CHOICES = {
    "Protocol": {6: "6 (TCP)", 17: "17 (UDP)"},
    "PSH Flag Count": {0: "0 (no PSH flag)", 1: "1 (PSH flag set)"},
}

# Display name: (key in metrics_knn.json, saved model file)
MODELS = {
    "k-NN": ("knn", "knn_model.joblib"),
    "Decision tree": ("decision_tree", "tree_model.joblib"),
    "Random Forest": ("random_forest", "forest_model.joblib"),
}


# =============================================================== DATA LOADERS
def read_json(name):
    path = config.RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def feature_columns():
    return json.loads((config.DATA_PROCESSED / "feature_columns.json").read_text())


@st.cache_data
def load_family_counts():
    train = pd.read_parquet(config.DATA_PROCESSED / "train.parquet", columns=["family"])
    test = pd.read_parquet(config.DATA_PROCESSED / "test.parquet", columns=["family"])
    return pd.concat([train, test])["family"].value_counts()


@st.cache_data
def load_test():
    cols = feature_columns() + ["Source IP", "family"]
    return pd.read_parquet(config.DATA_PROCESSED / "test.parquet", columns=cols)


@st.cache_resource
def load_models():
    bundles = {}
    for name, (_, filename) in MODELS.items():
        path = config.DATA_PROCESSED / filename
        if path.exists():
            bundle = joblib.load(path)
            if hasattr(bundle["model"], "n_jobs"):
                bundle["model"].n_jobs = 1  # one flow at a time: no parallel workers
            bundles[name] = bundle
    scaler = joblib.load(config.DATA_PROCESSED / "scaler.joblib")
    return bundles, scaler


def available_results(res):
    return {name: res[key] for name, (key, _) in MODELS.items() if key in res}


# =============================================================== UI HELPERS
def kpi(label, value, sub="", accent="#38bdf8"):
    st.markdown(
        f"<div class='kpi' style='border-left:4px solid {accent}'>"
        f"<div class='kl'>{label}</div><div class='kv'>{value}</div>"
        f"<div class='ks'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def section(text):
    st.markdown(f"<div class='section'>{text}</div>", unsafe_allow_html=True)


def plot_confusion(cm, labels, title):
    cm = np.array(cm)
    shares = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1627")
    ax.set_facecolor("#0e1627")
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(colors="#cbd5e1")
    ax.set_xlabel("Predicted family", color="#cbd5e1")
    ax.set_ylabel("True family", color="#cbd5e1")
    ax.set_title(title, color="#e6edf7")
    for s in ax.spines.values():
        s.set_color("#334155")
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if shares[i, j] > 0.5 else "#0b1220"
            ax.text(j, i, f"{shares[i, j]:.1%}\n({cm[i, j]:,})",
                    ha="center", va="center", fontsize=7, color=colour)
    fig.tight_layout()
    return fig


def family_donut(counts):
    df = counts.rename_axis("family").reset_index(name="rows")
    df["share"] = df["rows"] / df["rows"].sum()
    df["label"] = df["family"] + " (" + (df["share"] * 100).round(1).astype(str) + "%)"
    colors = [FAMILY_COLORS.get(f, "#38bdf8") for f in df["family"]]

    base = alt.Chart(df).encode(theta=alt.Theta("rows:Q", stack=True))
    arc = base.mark_arc(innerRadius=60, outerRadius=105).encode(
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=df["label"].tolist(), range=colors),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=[
            "family",
            alt.Tooltip("rows:Q", format=",", title="Rows"),
            alt.Tooltip("share:Q", format=".1%", title="Share"),
        ],
    )
    text = (
        base.transform_filter(alt.datum.share >= 0.05)
        .mark_text(radius=83, color="white", fontWeight="bold", size=13)
        .encode(text=alt.Text("share:Q", format=".1%"))
    )
    return (arc + text).properties(height=270)


def proba_bars(probs):
    df = pd.DataFrame({"family": list(probs), "probability": list(probs.values())})
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("family:N", sort=None, title=None),
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
            color=alt.Color(
                "family:N",
                scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
                legend=None,
            ),
            tooltip=["family", alt.Tooltip("probability:Q", format=".1%")],
        )
        .properties(height=150)
    )


def per_family_table(metrics):
    df = pd.DataFrame(metrics["per_family"]).T.rename(columns={"f1-score": "f1"})
    cfg = {
        c: st.column_config.ProgressColumn(c.capitalize(), min_value=0.0, max_value=1.0, format="%.3f")
        for c in df.columns
    }
    return df, cfg


def run_model(bundle, scaler, flow, cols):
    X = scaler.transform(flow[cols]).astype("float32")
    X = bundle["selector"].transform(X)
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    k = int(np.argmax(proba))
    probs = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    return str(model.classes_[k]), float(proba[k]), probs


def to_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


@st.cache_data
def feature_guide(features):
    test = load_test()
    rows = []
    for f in features:
        r = {"Feature": f, "Min": test[f].min(), "Max": test[f].max()}
        for fam, g in test.groupby("family")[f]:
            r[fam] = f"{g.quantile(0.05):,.2f} to {g.quantile(0.95):,.2f}"
        rows.append(r)
    return pd.DataFrame(rows).set_index("Feature")


def random_flow_position(test):
    """A random flow from a random family. The family is never shown before predicting."""
    family = np.random.choice(test["family"].unique())
    return int(np.random.choice(np.where(test["family"] == family)[0]))


# =============================================================== PAGE 1: OVERVIEW
def page_data():
    stats = read_json("preprocessing_stats.json")
    if stats is None:
        st.error("Run `python -m src.p2_preprocess.clean` first.")
        return
    res = read_json("metrics_knn.json")
    n_total = len(feature_columns())
    n_used = len(res["features_used"]) if res else "-"

    section("Dataset summary")
    k = st.columns(4)
    with k[0]:
        kpi("Raw rows", f"{stats['rows_raw']:,}", "4 CICIDS2017 files", "#38bdf8")
    with k[1]:
        kpi("Clean rows", f"{stats['rows_clean']:,}", "after cleaning", "#22c55e")
    with k[2]:
        kpi("Our sample", f"{stats['rows_sample']:,}", f"{stats['sample_fraction']:.1%} of clean rows", "#a78bfa")
    with k[3]:
        kpi("Features", f"{n_used} of {n_total}", "selected for the models", "#f59e0b")

    k = st.columns(4)
    with k[0]:
        kpi("Missing / inf rows removed", f"{stats['missing_removed']:,}", "dropped during cleaning", "#ef4444")
    with k[1]:
        kpi("Duplicates removed", f"{stats['duplicates_removed']:,}", "repeated rows", "#ef4444")
    with k[2]:
        kpi("Train / test split", "80 / 20", "same family share in both", "#38bdf8")
    with k[3]:
        kpi("Attack families", "4", "Benign, BruteForce, DoS, PortScan", "#22c55e")

    section("Data details")
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            st.markdown("##### Files used")
            files = pd.DataFrame(
                [{"File": n, "Rows": v["rows"], "Columns": v["columns"]}
                 for n, v in stats["files"].items()]
            )
            st.dataframe(files, hide_index=True, use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("##### Traffic by family")
            st.altair_chart(family_donut(load_family_counts()), use_container_width=True)

    with st.expander("Feature histograms"):
        pics = sorted((config.RESULTS / "figures").glob("hist_*.png"))
        grid = st.columns(3)
        for i, p in enumerate(pics):
            grid[i % 3].image(str(p))


# =============================================================== PAGE 2: MODELS
def page_models():
    res = read_json("metrics_knn.json")
    if res is None:
        st.error("Run `python -m src.p3_supervised.knn_classifier` first.")
        return
    labels = res.get("labels", ["Benign", "BruteForce", "DoS", "PortScan"])
    results = available_results(res)
    best = max(results, key=lambda n: results[n]["macro_f1"])

    section("Best model")
    k = st.columns(4)
    with k[0]:
        kpi("Best model", best, "highest macro F1", "#22c55e")
    with k[1]:
        kpi("Macro F1", f"{results[best]['macro_f1']:.4f}", "all families weighted equally", "#38bdf8")
    with k[2]:
        kpi("Accuracy", f"{results[best]['accuracy']:.2%}", "on the test set", "#a78bfa")
    with k[3]:
        kpi("k-NN setting", f"k = {res['knn']['k']}", f"{res['knn']['weights']} weighting", "#f59e0b")

    section("Test-set scores")
    rows = {"Always say Benign": res["majority_baseline"], **results}
    comp = pd.DataFrame(
        [{"model": n, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]} for n, m in rows.items()]
    ).set_index("model")
    with st.container(border=True):
        st.dataframe(
            comp,
            use_container_width=True,
            column_config={
                "macro_f1": st.column_config.ProgressColumn("Macro F1", min_value=0.0, max_value=1.0, format="%.4f"),
                "accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0.0, max_value=1.0, format="%.4f"),
            },
        )
        st.caption("Most traffic is Benign, so accuracy alone is misleading. "
                   "Macro F1 gives every family equal weight.")

    section("Per-family results")
    with st.container(border=True):
        tabs = st.tabs(list(results))
        for tab, (name, metrics) in zip(tabs, results.items()):
            with tab:
                c1, c2 = st.columns(2)
                with c1:
                    df, cfg = per_family_table(metrics)
                    st.dataframe(df, column_config=cfg, use_container_width=True)
                with c2:
                    if "confusion" in metrics:
                        st.pyplot(plot_confusion(metrics["confusion"], labels, name))

    if "tuning" in res:
        with st.expander("k-NN: choosing k (macro F1 on validation data)"):
            tune = pd.DataFrame(res["tuning"])
            pivot = tune.pivot(index="k", columns="weights", values="macro_f1")
            st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)
            st.success(f"Best: k = {res['knn']['k']}, weights = {res['knn']['weights']}")

    st.info(
        "Limitations: only 4 families because we used 4 files (no web attacks, bot or "
        "infiltration). The train/test split is random. The weakest family is BruteForce."
    )


# =============================================================== PAGE 3: LIVE DETECTION
def page_predict():
    res = read_json("metrics_knn.json")
    if res is None:
        st.error("Run the k-NN script first.")
        return

    chosen = res["features_used"]
    cols = feature_columns()
    test = load_test()
    bundles, scaler = load_models()
    guide = feature_guide(tuple(chosen))
    results = available_results(res)
    usable = [n for n in results if n in bundles]
    if not usable:
        st.error("No saved model files found. Re-run the training scripts.")
        return
    best_name = max(usable, key=lambda n: results[n]["macro_f1"])

    # A random flow is chosen once per session (refresh the browser page for a new one)
    if "pos" not in st.session_state:
        st.session_state.pos = random_flow_position(test)
    pos = st.session_state.pos
    row = test.iloc[pos]
    src_ip = row.get("Source IP", "unknown")

    section("Live detection")
    k = st.columns(3)
    with k[0]:
        kpi("Source IP", src_ip, "origin of this flow", "#a78bfa")
    with k[1]:
        kpi("Detector", best_name, "highest macro F1", "#22c55e")
    with k[2]:
        kpi("Detector macro F1", f"{results[best_name]['macro_f1']:.4f}", "on the test set", "#38bdf8")

    left, right = st.columns([3, 2])

    # ---- left: editable features
    with left:
        with st.container(border=True):
            st.markdown("##### Flow features")
            with st.form("flow_form"):
                inputs = {}
                grid = st.columns(3)
                for i, feat in enumerate(chosen):
                    with grid[i % 3]:
                        if feat in CHOICES:
                            options = list(CHOICES[feat])
                            nearest = min(options, key=lambda o: abs(o - float(row[feat])))
                            inputs[feat] = st.selectbox(
                                feat, options, index=options.index(nearest),
                                format_func=CHOICES[feat].get, key=f"{pos}_{feat}",
                                help="Allowed values: " + ", ".join(str(o) for o in options),
                            )
                        else:
                            inputs[feat] = st.text_input(
                                feat, value=f"{row[feat]:.4f}", key=f"{pos}_{feat}",
                                help=f"Range in the data: {guide.loc[feat, 'Min']:,.2f} to "
                                     f"{guide.loc[feat, 'Max']:,.2f}",
                            )
                submitted = st.form_submit_button("🔍 Analyze flow", type="primary")

            if submitted:
                values = {f: to_number(t) for f, t in inputs.items()}
                bad = [f for f, v in values.items() if v is None]
                if bad:
                    st.error("Enter plain numbers only. Check: " + ", ".join(bad))
                else:
                    outside = [f for f, v in values.items()
                               if f not in CHOICES
                               and not (guide.loc[f, "Min"] <= v <= guide.loc[f, "Max"])]
                    flow = row[cols].astype("float64").to_frame().T
                    for f, v in values.items():
                        flow[f] = v
                    pred, conf, probs = run_model(bundles[best_name], scaler, flow, cols)
                    st.session_state.result = {
                        "pos": pos, "pred": pred, "conf": conf,
                        "probs": probs, "outside": outside,
                    }

    # ---- right: result panel
    with right:
        with st.container(border=True):
            st.markdown("##### Detection result")
            result = st.session_state.get("result")
            if result is None or result["pos"] != pos:
                st.info("Edit any feature if you like, then press **Analyze flow**.")
            else:
                pred = result["pred"]
                colour = FAMILY_COLORS.get(pred, "#38bdf8")
                st.markdown(
                    f"<div class='verdict' style='border-color:{colour}; background:{colour}22'>"
                    f"<div class='vl'>PREDICTED FAMILY</div>"
                    f"<div class='vv' style='color:{colour}'>{pred}</div>"
                    f"<div class='vs'>{result['conf']:.0%} confident · {best_name}</div>"
                    f"<span class='pill' style='background:{colour}'>{ACTION_TAG.get(pred, 'REVIEW')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                target = f"<br><span style='color:#9aa4b2'>Target: {src_ip}</span>" if pred != "Benign" else ""
                st.markdown(
                    f"<div class='action'><div class='kl'>Autonomous action</div>"
                    f"<b>{ACTIONS.get(pred, 'Review manually')}</b>{target}</div>",
                    unsafe_allow_html=True,
                )
                st.altair_chart(proba_bars(result["probs"]), use_container_width=True)
                if result["outside"]:
                    st.warning("Outside the range seen in the data (prediction may be unreliable): "
                               + ", ".join(result["outside"]))

    # ---- helpers below
    with st.expander("Which features should I change? (typical values per family)"):
        st.caption("Each column shows the usual range (5th to 95th percentile) of that feature "
                   "for one family. To make a flow look like an attack, move the values into that range.")
        st.dataframe(guide.style.format({"Min": "{:,.2f}", "Max": "{:,.2f}"}), use_container_width=True)

    with st.expander("Response rules"):
        st.dataframe(
            pd.DataFrame([{"Predicted family": f, "Action": a, "Tag": ACTION_TAG[f]}
                          for f, a in ACTIONS.items()]),
            hide_index=True, use_container_width=True,
        )

# =============================================================== PAGE 4: SIMULATION
def page_simulation():
    res = read_json("metrics_knn.json")
    if res is None:
        st.error("Run the k-NN script first.")
        return

    cols = feature_columns()
    test = load_test()
    bundles, scaler = load_models()
    results = available_results(res)
    usable = [n for n in results if n in bundles]
    if not usable:
        st.error("No saved model files found. Re-run the training scripts.")
        return
    best_name = max(usable, key=lambda n: results[n]["macro_f1"])
    bundle = bundles[best_name]

    section("Autonomous response simulation")
    st.caption(f"The system analyses random flows with the {best_name} model and "
               "applies a response to each one, with no human in the loop.")

    c1, c2 = st.columns([3, 1])
    n = c1.slider("Number of flows to analyse", 20, 300, 100, step=20)
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    run = c2.button("▶ Run simulation", type="primary")

    if run:
        with st.spinner("Analysing traffic..."):
            sample = test.sample(n)
            X = scaler.transform(sample[cols]).astype("float32")
            X = bundle["selector"].transform(X)
            model = bundle["model"]
            proba = model.predict_proba(X)
            pred = model.classes_[proba.argmax(axis=1)]
            log = pd.DataFrame({
                "Source IP": sample["Source IP"].to_numpy(),
                "Predicted": pred,
                "Confidence": proba.max(axis=1),
                "Action": [ACTION_TAG.get(p, "REVIEW") for p in pred],
                "Actual": sample["family"].to_numpy(),
            })
            log["Correct"] = log["Predicted"] == log["Actual"]
            st.session_state.sim = log

    log = st.session_state.get("sim")
    if log is None:
        st.info("Press **Run simulation** to start.")
        return

    threats = log[log["Predicted"] != "Benign"]
    blocked = log[log["Action"].isin(["BLOCK", "RATE-LIMIT"])]
    missed = log[(log["Actual"] != "Benign") & (log["Predicted"] == "Benign")]

    k = st.columns(4)
    with k[0]:
        kpi("Flows analysed", f"{len(log):,}", "random test flows", "#38bdf8")
    with k[1]:
        kpi("Threats detected", f"{len(threats):,}", "predicted as an attack", "#f59e0b")
    with k[2]:
        kpi("IPs blocked / limited", f"{blocked['Source IP'].nunique():,}", "unique source IPs", "#ef4444")
    with k[3]:
        kpi("Attacks missed", f"{len(missed):,}", "predicted Benign but were attacks", "#a78bfa")

    section("Decision log")
    with st.container(border=True):
        st.dataframe(
            log,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.0%"),
            },
        )
        st.caption("'Actual' is the true label, shown only to check the system's decisions.")

    if len(blocked):
        with st.expander("Blocked / rate-limited IPs"):
            summary = (
                blocked.groupby("Source IP")
                .agg(
                    Flows=("Predicted", "size"),
                    Attacks=("Predicted", lambda s: ", ".join(f"{k} ({v})" for k, v in s.value_counts().items())),
                    Actions=("Action", lambda s: ", ".join(sorted(set(s)))),
                )
                .reset_index()
                .sort_values("Flows", ascending=False)
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)

# =============================================================== MAIN
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='hero'><div class='hero-t'>🛡️ SENTRIX</div>"
        "<div class='hero-s'>Autonomous Cyber-Defense &amp; Intrusion Detection System</div></div>",
        unsafe_allow_html=True,
    )

    pages = {
        "📊 Overview": page_data,
        "🤖 Models": page_models,
        "🚨 Live detection": page_predict,
        "🛰️ Simulation": page_simulation,
    }
    st.sidebar.markdown("### 🛡️ SENTRIX")
    choice = st.sidebar.radio("Navigate", list(pages), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Dataset: CICIDS2017 (4 files)")
    pages[choice]()


main()