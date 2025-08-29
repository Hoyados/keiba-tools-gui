import numpy as np
import pandas as pd
import argparse
import time
import io
import os
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO, ERROR

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import urllib.request

# 日本語フォントがない環境（Streamlit Cloud 等）でも表示できるようにフォールバック設定
plt.rcParams["font.family"] = [
    "Noto Sans JP",
    "Hiragino Sans",
    "Hiragino Kaku Gothic Pro",
    "Yu Gothic",
    "Meiryo",
    "IPAGothic",
    "DejaVu Sans",
]
plt.rcParams['axes.unicode_minus'] = False

_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
_FONT_PATH = os.path.join(_FONT_DIR, "NotoSansJP-Regular.otf")
_FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansJP-Regular.otf"

def ensure_noto_sans_jp_available(use_streamlit_cache: bool = False):
    """
    Noto Sans JP をダウンロードして Matplotlib に登録する。
    既に利用可能なら何もしない。Streamlit 環境ではキャッシュ可能。
    """
    def _do_install():
        try:
            # 既存のフォント探索
            for f in fm.fontManager.ttflist:
                name = getattr(f, 'name', '') or ''
                if 'Noto Sans JP' in name or 'NotoSansJP' in name:
                    plt.rcParams["font.family"] = ["Noto Sans JP", "DejaVu Sans"]
                    return True
        except Exception:
            pass

        try:
            os.makedirs(_FONT_DIR, exist_ok=True)
            if not os.path.exists(_FONT_PATH):
                with urllib.request.urlopen(_FONT_URL, timeout=30) as r:
                    data = r.read()
                with open(_FONT_PATH, "wb") as w:
                    w.write(data)
            fm.fontManager.addfont(_FONT_PATH)
            try:
                fm._rebuild()  # 一部バージョンで必要
            except Exception:
                pass
            plt.rcParams["font.family"] = ["Noto Sans JP", "DejaVu Sans"]
            return True
        except Exception as e:
            logger.warning(f"Noto Sans JP の準備に失敗しました: {e}")
            return False

    if use_streamlit_cache:
        try:
            import streamlit as st
            return st.cache_resource(show_spinner=False)(_do_install)()
        except Exception:
            return _do_install()
    else:
        return _do_install()

# 必須列（前処理に必要）
REQUIRED_COLUMNS = [
    "日付",
    "競馬場",
    "馬券種別",
    "購入金額",
    "倍率",
    "的中",
]

# ロガー設定（output ディレクトリがない場合は作成）
logger = getLogger(__name__)
logger.setLevel(INFO)

def _ensure_output_dir():
    try:
        os.makedirs("output", exist_ok=True)
    except Exception:
        pass

def _setup_logger_handlers():
    if any(isinstance(h, FileHandler) for h in logger.handlers):
        return
    console_handler = StreamHandler()
    console_handler.setLevel(ERROR)
    console_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    _ensure_output_dir()
    try:
        file_handler = FileHandler("output/log.txt", encoding="utf-8")
        file_handler.setLevel(INFO)
        file_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
    except Exception:
        # ファイルハンドラ作成に失敗した場合でもコンソールは残す
        pass

    logger.addHandler(console_handler)
    logger.propagate = False

_setup_logger_handlers()
start_time = time.perf_counter()
logger.info("処理開始")

# ファイル読み込み関数
def fileload(filename: str):
    try:
        df = pd.read_csv(f"data/{filename}.csv")
        logger.info(f"読み込みファイル: data/{filename}.csv")
        return df
    except Exception:
        logger.error("ファイルがありません")
        raise

def cleandata(df):
    """
    データのクリーニングを行う関数。
        - 欠損が多すぎる行の削除
        - 金額、倍率の整形、型変換（倍率データなしを-1.0倍にする）
        - 馬券種別の表記揺れ修正
        - 収支差列を作成
    引数: dataframe（元データ）
    戻り値: dataframe（クリーン済みデータ）
    """
    df = df.copy()
    df = df[df.isna().sum(axis = 1) < 7]
    df["購入金額"] = df["購入金額"].replace("[¥,]", "", regex = True).astype(int)
    df["倍率"] = df["倍率"].replace("[――—]", None, regex = True).astype(float)
    df["倍率"] = df["倍率"].fillna(-1.0)
    df["馬券種別"] = df["馬券種別"].replace("[ ]", "", regex = True)
    df["馬券種別"] = df["馬券種別"].replace("[3 ３]", "三", regex = True)
    df["払い戻し"] = np.where(df['的中'],df["購入金額"] * df["倍率"],0)
    df["収支差"] = df["払い戻し"].astype(float) - df["購入金額"]
    return df

def add_feature(df):
    """
    データに特徴量を追加する
    - 日付を月単位分類
    - 倍率を1倍区切りで分類
    - 回収率を追加
    - 中央 / 地方の区分けを追加
    """
    df["日付"] = df["日付"].astype(str).str.strip()
    df["日付"] = pd.to_datetime(df["日付"])
    df["月"] = df["日付"].dt.to_period("M")

    df["倍率帯"] = pd.cut(df["倍率"], bins=[0, 2.5, 4, 8, 100], labels=["1〜2.5倍", "2.6〜4倍", "4〜8倍", "8倍以上"])

    df.loc[df["競馬場"].isin(["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]), "中央地方"] = "中央"
    df.loc[df["競馬場"].isin(["門別","盛岡","水沢","浦和","船橋","大井","川崎","金沢","名古屋","笠松","園田","姫路","高知","佐賀"]), "中央地方"] = "地方"


    return df

def summarize_by (df, key_col, target_col, agg_func, sort = True):
    """
    汎用集計関数
    引数: 
    - dataframe(クリーン済み)
    - 集計軸
    - 集計対象
    - 集計方式（平均mean, 総和sumなど）
    """
    summary = df.groupby(key_col)[target_col].agg(agg_func)
    if sort:
        summary = summary.sort_index()
    return summary

def heatmap_data (df, index, columns, values, aggfunc = "count"):
    """
    ピボットテーブルにより、ヒートマップ用dfを作成する
    引数
    - index: 縦
    - columns: 横
    - values: 値
    - aggfunc: 関数（平均mean、総和sum、回数countなど）
    """
    pivot_data = df.pivot_table(index = index, columns = columns, values = values, aggfunc = aggfunc, observed = False)
    return pivot_data

# 2軸グラフ描画関数
def plotgraph (temp1, temp2, xlabel, y1label, y2label):

    fig, ax1 = plt.subplots()

# 左Y軸
    ax1.bar(temp1.index.astype(str), temp1, label = y1label, color = "blue")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color = "blue")
    ax1.tick_params(axis="y")

# 右Y軸
    ax2 = ax1.twinx()
    ax2.plot(temp2.index.astype(str), temp2, marker="o", label = y2label, color = "red")
    ax2.set_ylabel(y2label, color = "red")
    ax2.tick_params(axis="y")

# タイトルとレイアウト
    plt.title(xlabel + "ごとの" + y1label + "と" + y2label)
    fig.tight_layout()
    plt.savefig("output/" + xlabel +".png")

# 棒グラフ描画関数
def bargraph (temp1, label):
    fig, ax = plt.subplots(figsize = (6,6))
    ax.bar(temp1.index, temp1.values, width = 0.1)
    # ax.set_xlim(1.0, 4.0)
    xlabel = f"{label}"
    fig.tight_layout()
    plt.savefig("output/" + xlabel +".png")

# ヒートマップ描画関数
def heatmapgraph (pivot_table, label1, label2, value, max = 100, min = 0):
    fig, ax = plt.subplots(figsize = (8, 8))
    sns.heatmap(pivot_table, annot = True, cmap = "coolwarm", ax = ax, fmt = ".0f", square = True, vmax = max, vmin = min, linewidths = 0.5)
    plt.title(f"{label1}と{label2}による{value}ヒートマップ")
    fig.tight_layout()
    plt.savefig(f"output/{label1}x{label2}_{value}.png")

# ===== Streamlit 用ユーティリティ =====
def validate_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

def compute_overall_metrics(df: pd.DataFrame):
    total_buy = float(df["購入金額"].sum())
    total_return = float(df["払い戻し"].sum())
    profit = total_return - total_buy
    recovery = (total_return / total_buy * 100.0) if total_buy > 0 else 0.0
    return {
        "総購入金額": total_buy,
        "総払い戻し": total_return,
        "収支差": profit,
        "回収率": recovery,
    }

def available_group_columns(df: pd.DataFrame):
    candidates = []
    for col in df.columns:
        if str(col) in ["購入金額", "払い戻し", "収支差", "倍率", "的中"]:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_period_dtype(df[col]):
            candidates.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
        else:
            # 月や倍率帯、中央地方などは上の分岐でも拾える
            pass
    # よく使う列を前に
    priority = ["月", "倍率帯", "馬券種別", "中央地方", "競馬場"]
    ordered = [c for c in priority if c in candidates] + [c for c in candidates if c not in priority]
    return ordered

def aggregate_series(df: pd.DataFrame, group_col: str, metric_key: str) -> pd.Series:
    if metric_key == "的中率(%)":
        s = df.groupby(group_col)["的中"].mean() * 100.0
    elif metric_key == "収支差(円)":
        s = df.groupby(group_col)["収支差"].sum()
    elif metric_key == "払い戻し(円)":
        s = df.groupby(group_col)["払い戻し"].sum()
    elif metric_key == "購入金額(円)":
        s = df.groupby(group_col)["購入金額"].sum()
    elif metric_key == "購入回数":
        s = df.groupby(group_col)["的中"].count()
    elif metric_key == "的中回数":
        s = df.groupby(group_col)["的中"].sum()
    elif metric_key == "回収率(%)":
        r = df.groupby(group_col)["払い戻し"].sum()
        b = df.groupby(group_col)["購入金額"].sum()
        s = (r / b.replace(0, np.nan)) * 100.0
    elif metric_key == "一回あたりの収支(円)":
        r = df.groupby(group_col)["払い戻し"].sum()
        b = df.groupby(group_col)["購入金額"].sum()
        c = df.groupby(group_col)["的中"].count()
        s = (r - b) / c.replace(0, np.nan)
    else:
        raise ValueError("未知のメトリクスです")
    return s

def aggregate_pivot(df: pd.DataFrame, index: str, columns: str, metric_key: str) -> pd.DataFrame:
    if metric_key == "的中率(%)":
        pv = df.pivot_table(index=index, columns=columns, values="的中", aggfunc="mean", observed=False) * 100.0
    elif metric_key == "収支差(円)":
        pv = df.pivot_table(index=index, columns=columns, values="収支差", aggfunc="sum", observed=False)
    elif metric_key == "払い戻し(円)":
        pv = df.pivot_table(index=index, columns=columns, values="払い戻し", aggfunc="sum", observed=False)
    elif metric_key == "購入金額(円)":
        pv = df.pivot_table(index=index, columns=columns, values="購入金額", aggfunc="sum", observed=False)
    elif metric_key == "購入回数":
        pv = df.pivot_table(index=index, columns=columns, values="的中", aggfunc="count", observed=False)
    elif metric_key == "的中回数":
        pv = df.pivot_table(index=index, columns=columns, values="的中", aggfunc="sum", observed=False)
    elif metric_key == "回収率(%)":
        r = df.pivot_table(index=index, columns=columns, values="払い戻し", aggfunc="sum", observed=False)
        b = df.pivot_table(index=index, columns=columns, values="購入金額", aggfunc="sum", observed=False)
        pv = (r / b.replace(0, np.nan)) * 100.0
    elif metric_key == "一回あたりの収支(円)":
        r = df.pivot_table(index=index, columns=columns, values="払い戻し", aggfunc="sum", observed=False)
        b = df.pivot_table(index=index, columns=columns, values="購入金額", aggfunc="sum", observed=False)
        c = df.pivot_table(index=index, columns=columns, values="的中", aggfunc="count", observed=False)
        pv = (r - b) / c.replace(0, np.nan)
    else:
        raise ValueError("未知のメトリクスです")
    return pv

def fig_bar(series: pd.Series, title: str = ""):
    fig, ax = plt.subplots(figsize=(8, 4))
    series = series.sort_index()
    ax.bar(series.index.astype(str), series.values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(series.index.name if series.index.name else "")
    ax.set_ylabel("値")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def fig_line(series: pd.Series, title: str = ""):
    fig, ax = plt.subplots(figsize=(8, 4))
    series = series.sort_index()
    ax.plot(series.index.astype(str), series.values, marker="o", color="#F58518")
    ax.set_title(title)
    ax.set_xlabel(series.index.name if series.index.name else "")
    ax.set_ylabel("値")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def fig_heatmap(pivot_df: pd.DataFrame, title: str = "", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="coolwarm", vmin=vmin, vmax=vmax, cbar=True, linewidths=0.5, ax=ax, square=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig

# Plotly 版（Streamlit ではこちらを利用）
def fig_bar_plotly(series: pd.Series, title: str = ""):
    import plotly.express as px
    s = series.sort_index()
    fig = px.bar(x=s.index.astype(str), y=s.values, title=title, labels={"x": s.index.name or "", "y": "値"})
    fig.update_layout(template="plotly_white", font=dict(family="Noto Sans JP"))
    return fig

def fig_line_plotly(series: pd.Series, title: str = ""):
    import plotly.express as px
    s = series.sort_index()
    fig = px.line(x=s.index.astype(str), y=s.values, title=title, markers=True, labels={"x": s.index.name or "", "y": "値"})
    fig.update_layout(template="plotly_white", font=dict(family="Noto Sans JP"))
    return fig

def fig_heatmap_plotly(pivot_df: pd.DataFrame, title: str = "", vmin=None, vmax=None):
    import plotly.express as px
    # x は列、y は行
    x = pivot_df.columns.astype(str)
    y = pivot_df.index.astype(str)
    fig = px.imshow(
        pivot_df.values,
        x=x,
        y=y,
        color_continuous_scale="RdBu",
        zmin=vmin,
        zmax=vmax,
        text_auto=True,
        aspect="auto",
        title=title,
    )
    fig.update_layout(template="plotly_white", font=dict(family="Noto Sans JP"))
    return fig

def run_streamlit_app():
    try:
        import streamlit as st
    except Exception:
        raise SystemExit("Streamlit がインストールされていません。`pip install streamlit` を実行してください。")

    st.set_page_config(page_title="競馬ツール GUI", layout="wide")
    # UI 全体も Noto Sans JP を利用（Matplotlib には影響しませんが UI の統一感向上）
    st.markdown(
        "<link href='https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap' rel='stylesheet'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<style>html, body, [class*='css'] { font-family: 'Noto Sans JP', sans-serif !important; }</style>",
        unsafe_allow_html=True,
    )
    st.title("競馬結果分析ツール (Streamlit)")

    # 日本語フォントの用意（クラウド環境向け）
    ensure_noto_sans_jp_available(use_streamlit_cache=True)

    st.sidebar.header("CSV 読み込み")
    uploaded = st.sidebar.file_uploader("CSVファイルを選択", type=["csv"])

    df = None
    if uploaded is not None:
        # 文字コードのフォールバック
        try:
            df = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            df = pd.read_csv(uploaded, encoding="cp932")
        except Exception as e:
            st.error(f"CSV読み込み時にエラー: {e}")
            df = None

    if df is None:
        st.info("左のサイドバーからCSVをアップロードしてください。")
        st.stop()

    # 必須列チェック
    missing = validate_required_columns(df)
    if missing:
        st.error("次の必須列がありません: " + ", ".join(missing))
        st.stop()

    # 既存の整形ロジックを適用（キャッシュ）
    @st.cache_data(show_spinner=False)
    def _process(df_input: pd.DataFrame) -> pd.DataFrame:
        df2 = cleandata(df_input)
        df3 = add_feature(df2)
        return df3

    try:
        df = _process(df)
    except Exception as e:
        st.error(f"データ整形に失敗しました: {e}")
        st.stop()

    st.subheader("ファイルプレビュー (先頭10行)")
    st.dataframe(df.head(10))

    st.subheader("集計指標")
    metrics = compute_overall_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総購入金額", f"{metrics['総購入金額']:,.0f} 円")
    c2.metric("総払い戻し", f"{metrics['総払い戻し']:,.0f} 円")
    c3.metric("収支差", f"{metrics['収支差']:,.0f} 円")
    c4.metric("回収率", f"{metrics['回収率']:.1f} %")

    st.subheader("グラフ作成")
    group_cols = available_group_columns(df)
    if len(group_cols) == 0:
        st.warning("グルーピング可能な列が見つかりません。")
        st.stop()

    left, right = st.columns([1, 1])
    with left:
        x_col = st.selectbox("横軸 (グループ化列)", options=group_cols, index=0, key="x_col")
        y_col = st.selectbox("第二軸 (オプション: 列)", options=["(なし)"] + [c for c in group_cols if c != x_col], index=0, key="y_col")

    metric_options = [
        "的中率(%)",
        "回収率(%)",
        "収支差(円)",
        "一回あたりの収支(円)",
        "払い戻し(円)",
        "購入金額(円)",
        "購入回数",
        "的中回数",
    ]
    with right:
        metric_key = st.selectbox("集計値", options=metric_options, index=0, key="metric_key")

    two_dim = y_col != "(なし)"
    if two_dim:
        graph_types_available = ["ヒートマップ"]
    else:
        graph_types_available = ["棒グラフ", "折れ線"]

    graph_type = st.radio("グラフタイプ", options=graph_types_available, horizontal=True, help=("第二軸を選ぶとヒートマップのみ表示可能です"))

    fig = None
    try:
        if two_dim:
            pv = aggregate_pivot(df, index=x_col, columns=y_col, metric_key=metric_key)
            title = f"{x_col} × {y_col} の {metric_key}"
            # 見やすさのために vmax/vmin を自動推定（ヒートマップのみ）
            vmin, vmax = None, None
            if metric_key.endswith("%)"):
                vmin, vmax = 0, 100
            fig = fig_heatmap_plotly(pv, title=title, vmin=vmin, vmax=vmax)
        else:
            s = aggregate_series(df, group_col=x_col, metric_key=metric_key)
            title = f"{x_col} ごとの {metric_key}"
            if graph_type == "棒グラフ":
                fig = fig_bar_plotly(s, title=title)
            else:
                fig = fig_line_plotly(s, title=title)
    except Exception as e:
        st.error(f"グラフ作成時にエラーが発生しました: {e}")
        return

    if fig is not None:
    st.plotly_chart(fig, use_container_width=True)

    import plotly.io as pio
    import io, importlib.util, traceback

    # kaleido の見つかり方を可視化（名前衝突や未導入を判定）
    spec = importlib.util.find_spec("kaleido")
    if not spec:
        st.info("PNG 書き出しには 'kaleido' が必要です。requirements.txt に 'kaleido' を追加して再デプロイしてください。")
        buf = None
    else:
        # ここで実パスも確認できると安心（必要なら表示）
        # st.sidebar.write("kaleido at:", spec.origin)

        try:
            # 明示的に engine='kaleido' を指定
            png_bytes = pio.to_image(fig, format="png", scale=2, engine="kaleido")
            buf = io.BytesIO(png_bytes)
        except ModuleNotFoundError as e:
            st.info("PNG 書き出しには 'kaleido' が必要です。requirements.txt に 'kaleido' を追加して再デプロイしてください。")
            buf = None
        except Exception as e:
            # kaleido 以外の原因（フォント/互換/権限など）は内容を表示
            st.warning("PNG 書き出しに失敗しました（kaleido 以外の要因の可能性があります）。")
            st.exception(e)  # 具体的なエラー内容を表示
            buf = None

    default_name = f"{x_col}{'x' + y_col if two_dim else ''}_{metric_key}.png".replace("/", "-")
    if buf is not None:
        st.download_button(
            label="グラフを保存 (PNG)",
            data=buf,
            file_name=default_name,
            mime="image/png",
        )

def main_cli(filename: str = "競馬-結果リスト"):
    df = fileload(filename)
    df = cleandata(df)
    df = add_feature(df)
    _ensure_output_dir()
    # CLI 実行時も可能ならフォントを用意
    ensure_noto_sans_jp_available(use_streamlit_cache=False)
    df.to_csv("output/cleaned-keibadata.csv", index = False)

    # 全データの集計
    hitrate = summarize_by(df, "馬券種別", "的中", "mean", False) * 100
    returnrate = summarize_by(df, "馬券種別", "収支差", "sum", False)
    plotgraph (hitrate, returnrate, "馬券種別", "的中率(%)", "収支差(円)")
    hitrate = summarize_by(df, "倍率帯", "的中", "mean", False) * 100
    returnrate = summarize_by(df, "倍率帯", "収支差", "sum", False)
    plotgraph (hitrate, returnrate, "倍率帯", "的中率(%)", "収支差(円)")
    hitrate = summarize_by(df, "月", "的中", "mean", True) * 100
    returnrate = summarize_by(df, "月", "収支差", "sum", True)
    plotgraph (hitrate, returnrate, "月", "的中率(%)", "収支差(円)")
    pivot_table = heatmap_data(df, "馬券種別", "倍率帯", "収支差", "sum")
    heatmapgraph (pivot_table, "馬券種別", "倍率帯", "収支", 2500, -2500)
    pivot_table = heatmap_data(df, "馬券種別", "倍率帯", "的中", "mean") * 100
    heatmapgraph (pivot_table, "馬券種別", "倍率帯", "的中率", 100, 0)
    pivot_table = heatmap_data(df, "馬券種別", "倍率帯", "的中", "count")
    heatmapgraph (pivot_table, "馬券種別", "倍率帯", "購入回数", 40, 0)
    pivot_table = heatmap_data(df, "馬券種別", "倍率帯", "的中", "sum")
    heatmapgraph (pivot_table, "馬券種別", "倍率帯", "的中回数", 40, 0)
    returnmoney = heatmap_data(df, "馬券種別", "倍率帯", "払い戻し", "sum")
    buymoney = heatmap_data(df, "馬券種別", "倍率帯", "購入金額", "sum")
    counts = heatmap_data(df, "馬券種別", "倍率帯", "的中", "count")
    pivot_table = returnmoney / buymoney * 100
    heatmapgraph (pivot_table, "馬券種別", "倍率帯", "回収率", 200, 0)
    pivot_table = (returnmoney - buymoney) / counts
    heatmapgraph (pivot_table, "馬券種別", "倍率帯", "一回あたりの収支", 200, -200)
    temp = summarize_by (df, "中央地方", "収支差", "sum", False)
    bargraph (temp, "中央_地方の倍率分布")

    # 複勝、ワイドのみの集計
    temp_df = df[df["馬券種別"].isin(["複勝"])]
    temp = summarize_by (temp_df, "倍率", "的中", "count")
    bargraph (temp, "複勝の倍率分布")
    temp_df = df[df["馬券種別"].isin(["ワイド"])]
    temp = summarize_by (temp_df, "倍率", "的中", "count")
    bargraph (temp, "ワイドの倍率分布")
    temp_df = df[df["馬券種別"].isin(["単勝"])]
    temp = summarize_by (temp_df, "倍率", "的中", "count")
    bargraph (temp, "単勝の倍率分布")

    # 単勝、ワイドのみの集計
    temp_df = df[df["馬券種別"].isin(["単勝", "ワイド"])]
    hitrate = summarize_by(temp_df, "馬券種別", "的中", "mean", False) * 100
    returnrate = summarize_by(temp_df, "馬券種別", "収支差", "sum", False)
    plotgraph (hitrate, returnrate, "単勝・ワイド", "的中率(%)", "収支差(円)")
    hitrate = summarize_by(temp_df, "倍率帯", "的中", "mean", False) * 100
    returnrate = summarize_by(temp_df, "倍率帯", "収支差", "sum", False)
    plotgraph (hitrate, returnrate, "倍率帯(単勝・ワイドのみ)", "的中率(%)", "収支差(円)")
    hitrate = summarize_by(temp_df, "月", "的中", "mean", True) * 100
    returnrate = summarize_by(temp_df, "月", "収支差", "sum", True)
    plotgraph (hitrate, returnrate, "月(単勝・ワイドのみ)", "的中率(%)", "収支差(円)")
    pivot_table = heatmap_data(temp_df, "馬券種別", "倍率帯", "収支差", "sum")
    heatmapgraph (pivot_table, "単勝・ワイド", "倍率帯", "収支", 2500, -2500)
    pivot_table = heatmap_data(temp_df, "馬券種別", "倍率帯", "的中", "mean") * 100
    heatmapgraph (pivot_table, "単勝・ワイド", "倍率帯", "的中率", 100, 0)
    pivot_table = heatmap_data(temp_df, "馬券種別", "倍率帯", "的中", "count")
    heatmapgraph (pivot_table, "単勝・ワイド", "倍率帯", "購入回数", 40, 0)
    pivot_table = heatmap_data(temp_df, "馬券種別", "倍率帯", "的中", "sum")
    heatmapgraph (pivot_table, "単勝・ワイド", "倍率帯", "的中回数", 40, 0)
    returnmoney = heatmap_data(temp_df, "馬券種別", "倍率帯", "払い戻し", "sum")
    buymoney = heatmap_data(temp_df, "馬券種別", "倍率帯", "購入金額", "sum")
    counts = heatmap_data(temp_df, "馬券種別", "倍率帯", "的中", "count")
    pivot_table = returnmoney / buymoney * 100
    heatmapgraph (pivot_table, "単勝・ワイド", "倍率帯", "回収率", 200, 0)
    pivot_table = (returnmoney - buymoney) / counts
    heatmapgraph (pivot_table, "単勝・ワイド", "倍率帯", "一回あたりの収支", 200, -200)

if __name__ == "__main__":
    # Streamlit 実行下かどうかを判定
    import sys
    running_under_streamlit = "streamlit" in sys.modules

    if running_under_streamlit:
        run_streamlit_app()
    else:
        # Streamlit 前提のため、通常実行時は案内を表示
        print("このアプリは Streamlit での実行を前提としています。\n")
        print("起動コマンド:  streamlit run src/keiba-tools-gui.py\n")
        # ただし、必要なら --cli で従来CLIを実行可能
        parser = argparse.ArgumentParser(description="競馬結果分析 (Streamlit推奨)" )
        parser.add_argument("-f", "--filename", type=str, default="競馬-結果リスト")
        parser.add_argument("--cli", action="store_true", help="従来のCLI処理を実行")
        args = parser.parse_args()
        if args.cli:
            try:
                main_cli(args.filename)
            finally:
                end_time = time.perf_counter()
                logger.info(f"処理終了: {round(end_time - start_time, 2)}秒")
