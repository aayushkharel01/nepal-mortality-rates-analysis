from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
AUTHOR = "Aayush Kharel"

YEARS = list(range(2000, 2024))  # 2000..2023 inclusive

# Snapshot frequency (2 = every 2 years, 1 = every year, 3 = every 3 years)
SNAPSHOT_STEP = 2

TOPN_SNAPSHOT = 20
TOPN_TABLE = 30

# Put your file inside: ~/Documents/Nepal_Data/data/
DATA_FILE = "nepal_analysis_2000_2023.csv"

# “Deep story” causes
FOCUS_CAUSES = ["Falls", "Self-harm", "Road injuries", "Drowning", "Interpersonal violence"]

# “Merged snapshots” plot: track top K causes over time
TOPK_TRACK = 20


# =========================
# Paths (works inside Nepal_Data)
# =========================
ROOT = Path(__file__).resolve().parents[1]  # .../Nepal_Data
csv_path = ROOT / "data" / DATA_FILE
fig_dir = ROOT / "figures"
out_dir = ROOT / "outputs"
fig_dir.mkdir(exist_ok=True)
out_dir.mkdir(exist_ok=True)


# =========================
# Helpers
# =========================
def add_signature(fig):
    fig.text(0.99, 0.01, AUTHOR, ha="right", va="bottom", fontsize=8, alpha=0.6)

def save_barh(top, title, outpath, xlabel="Deaths (number)"):
    fig_h = max(6, 0.45 * len(top) + 2)
    fig = plt.figure(figsize=(12, fig_h))
    plt.barh(top["cause_name"], top["val"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    add_signature(fig)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved:", outpath.name)


# =========================
# Load + clean
# =========================
if not csv_path.exists():
    raise FileNotFoundError(
        f"CSV not found at:\n{csv_path}\n\n"
        f"Put your file here:\n{ROOT/'data'/'nepal_analysis_2000_2023.csv'}"
    )

raw = pd.read_csv(csv_path)

required = {"location_name", "year", "cause_name", "val"}
missing = required - set(raw.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Found: {list(raw.columns)}")

df = raw[["location_name", "year", "cause_name", "val"]].copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["val"] = pd.to_numeric(df["val"], errors="coerce")
df = df.dropna(subset=["location_name", "year", "cause_name", "val"])
df["year"] = df["year"].astype(int)

# Keep Nepal rows only (safe)
df = df[df["location_name"].astype(str).str.contains("Nepal", case=False, na=False)].copy()

# Keep year range
df = df[df["year"].between(min(YEARS), max(YEARS))].copy()

years_present = sorted(df["year"].unique().tolist())
print("Loaded:", csv_path)
print("Rows:", len(df))
print("Years present:", years_present)

if not years_present:
    raise ValueError("No Nepal data found after filtering. Check your file.")

# totals + share
totals = df.groupby("year", as_index=False)["val"].sum().rename(columns={"val": "total_deaths"})
df = df.merge(totals, on="year", how="left")
df["share"] = df["val"] / df["total_deaths"]

df.to_csv(out_dir / "nepal_cleaned_with_share_2000_2023.csv", index=False)
print("Saved: nepal_cleaned_with_share_2000_2023.csv")


# =========================
# 1) Snapshots (every 2 years by default)
# =========================
snapshot_years = [y for y in YEARS if (y in years_present) and ((y - YEARS[0]) % SNAPSHOT_STEP == 0)]
if 2023 in years_present and 2023 not in snapshot_years:
    snapshot_years.append(2023)
snapshot_years = sorted(snapshot_years)

print("Snapshot years:", snapshot_years)

all_snapshot_tables = []

for y in snapshot_years:
    dy = df[df["year"] == y].copy()
    if dy.empty:
        continue

    # Top N snapshot plot
    top = dy.sort_values("val", ascending=False).head(TOPN_SNAPSHOT).sort_values("val")
    save_barh(
        top,
        title=f"Nepal — Top {TOPN_SNAPSHOT} causes ({y})",
        outpath=fig_dir / f"nepal_top{TOPN_SNAPSHOT}_{y}.png"
    )

    # Top 30 table
    t = dy.sort_values("val", ascending=False).head(TOPN_TABLE).copy()
    t.insert(0, "rank", range(1, len(t) + 1))
    t.to_csv(out_dir / f"nepal_top{TOPN_TABLE}_causes_{y}.csv", index=False)
    all_snapshot_tables.append(t)

if all_snapshot_tables:
    pd.concat(all_snapshot_tables, ignore_index=True).to_csv(
        out_dir / "nepal_snapshots_top30_all_years.csv", index=False
    )
    print("Saved: nepal_snapshots_top30_all_years.csv")


# =========================
# 2) “Merged snapshots”: track top-K causes over time (share)
# =========================
top_causes_overall = (
    df.groupby("cause_name", as_index=False)["val"].sum()
      .sort_values("val", ascending=False)
      .head(TOPK_TRACK)["cause_name"]
      .tolist()
)

track = df[df["cause_name"].isin(top_causes_overall)].copy()
pivot_share = (
    track.pivot_table(index="year", columns="cause_name", values="share", aggfunc="sum")
         .reindex(YEARS)
         .fillna(0)
)

fig = plt.figure(figsize=(14, 8))
for cause in pivot_share.columns:
    plt.plot(pivot_share.index, pivot_share[cause], linewidth=1, label=cause)

plt.title(f"Nepal — Top {TOPK_TRACK} causes tracked over time (share of deaths), 2000–2023")
plt.xlabel("Year")
plt.ylabel("Share of all deaths")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
add_signature(fig)

fig.savefig(fig_dir / f"nepal_top{TOPK_TRACK}_causes_share_trends_2000_2023.png", dpi=200)
plt.close(fig)
print("Saved: nepal_top20_causes_share_trends_2000_2023.png")

pivot_share.reset_index().to_csv(out_dir / "nepal_top_causes_share_trends_2000_2023.csv", index=False)
print("Saved: nepal_top_causes_share_trends_2000_2023.csv")


# =========================
# 3) Deep story: Falls vs Self-harm (share + deaths)
# =========================
focus = df[df["cause_name"].isin(FOCUS_CAUSES)].copy()
focus.to_csv(out_dir / "nepal_focus_injuries_2000_2023.csv", index=False)
print("Saved: nepal_focus_injuries_2000_2023.csv")

def plot_focus(metric="share"):
    pivot = (
        focus.pivot_table(index="year", columns="cause_name", values=metric, aggfunc="sum")
             .reindex(YEARS)
             .fillna(0)
    )

    fig = plt.figure(figsize=(12, 6))
    for cause in pivot.columns:
        plt.plot(pivot.index, pivot[cause], marker="o", label=cause)

    ylab = "Share of all deaths" if metric == "share" else "Deaths (number)"
    plt.title(f"Nepal — Falls vs Self-harm (and other injuries), 2000–2023 — {ylab}")
    plt.xlabel("Year")
    plt.ylabel(ylab)
    plt.legend(ncol=2)
    plt.tight_layout()
    add_signature(fig)

    out = fig_dir / f"nepal_focus_{metric}_2000_2023.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out.name)

plot_focus("share")
plot_focus("val")

print("\nDONE. Check:")
print("Figures:", fig_dir)
print("Outputs:", out_dir)
