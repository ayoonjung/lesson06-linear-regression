import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 한 해의 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온 추세선으로 원하는 해의 기온을 예상해 봅니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly_mean() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 2025년 이후 데이터 제외

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만인 해 제외
    return yearly.reset_index(drop=True)


try:
    yearly = load_yearly_mean()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("회귀 직선을 만들기에 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(x.min())
end_year = int(x.max())

# 최근 20년만으로 구한 회귀 (끝 연도 포함 20년 창)
RECENT_YEARS = 20
recent = yearly[yearly["연도"] > end_year - RECENT_YEARS]
recent_n = len(recent)
recent_start = int(recent["연도"].min())
recent_end = int(recent["연도"].max())
if recent_n >= 3:
    rx = recent["연도"].to_numpy(dtype=float)
    ry = recent["연평균기온"].to_numpy(dtype=float)
    recent_slope, recent_intercept = np.polyfit(rx, ry, 1)
    recent_r = float(np.corrcoef(rx, ry)[0, 1])
else:
    recent_slope = recent_intercept = recent_r = None

slope_100 = slope * 100  # 전체 기간: 100년당 기온 변화

# ---- 슬라이더와 예상 기온 ----
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">{year}년의 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.2;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- 데이터 정보 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")

st.write(
    f"사용한 데이터: {start_year}년부터 {end_year}년까지, 관측일이 {MIN_DAYS}일 이상인 "
    f"{n_years}개 해 (기준 기간: {LAST_YEAR}년까지). "
    f"회귀 직선: 기온 = {slope:.4f} × 연도 + {intercept:.2f}"
)

# ---- 기울기: 100년에 몇 도 오르는가 ----
st.divider()
st.markdown(
    f"""
    <div style="text-align:center; padding:0.6rem 0 1rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">전체 기간({start_year}~{end_year}년) 추세</div>
        <div style="font-size:3.5rem; font-weight:800; line-height:1.2;">
            100년에 {slope_100:+.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def slope_card(title, period, count, value_100, corr, color):
    return f"""
    <div style="border:2px solid {color}; border-radius:12px; padding:1rem; text-align:center;">
        <div style="font-size:1.05rem; font-weight:700;">{title}</div>
        <div style="font-size:0.9rem; opacity:0.7;">{period} · {count}개 해</div>
        <div style="font-size:2.4rem; font-weight:800; line-height:1.3;">{value_100:+.2f} ℃</div>
        <div style="font-size:0.9rem; opacity:0.7;">100년당 · 상관계수 r = {corr:.3f}</div>
    </div>
    """


st.subheader("기울기 비교: 전체 기간 vs 최근 20년")
col_all, col_recent = st.columns(2)
with col_all:
    st.markdown(
        slope_card("전체 기간", f"{start_year}~{end_year}년", n_years, slope_100, r, "#1f77b4"),
        unsafe_allow_html=True,
    )
with col_recent:
    if recent_slope is None:
        st.warning("최근 20년 구간에 데이터가 부족해요.")
    else:
        st.markdown(
            slope_card(
                "최근 20년",
                f"{recent_start}~{recent_end}년",
                recent_n,
                recent_slope * 100,
                recent_r,
                "crimson",
            ),
            unsafe_allow_html=True,
        )

if recent_slope is not None:
    diff = (recent_slope - slope) * 100
    if slope > 0 and recent_slope > 0:
        ratio_text = f"전체 기간의 약 {recent_slope / slope:.1f}배예요. "
    else:
        ratio_text = ""
    st.write(
        f"최근 {RECENT_YEARS}년의 기울기는 100년당 {recent_slope * 100:+.2f} ℃로, "
        f"전체 기간보다 {diff:+.2f} ℃ 차이가 나요. {ratio_text}"
        "짧은 구간은 해마다의 들쭉날쭉함에 크게 흔들리니 참고용으로 보세요."
    )

# ---- 산점도 + 회귀 직선 ----
line_x0 = min(start_year, year)
line_x1 = max(end_year, year)
line_x = np.array([line_x0, line_x1], dtype=float)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} ℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=slope * line_x + intercept,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="crimson"),
    )
)
if recent_slope is not None:
    recent_line_x = np.array([recent_start, recent_end], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=recent_line_x,
            y=recent_slope * recent_line_x + recent_intercept,
            mode="lines",
            name="최근 20년 직선",
            line=dict(width=3, color="royalblue", dash="dash"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=14, color="gold", symbol="star", line=dict(width=1, color="black")),
        hovertemplate="%{x}년 예상<br>%{y:.2f} ℃<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온 산점도와 회귀 직선 (상관계수 r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, use_container_width=True)

if year < start_year or year > end_year:
    st.info(
        f"{year}년은 데이터 범위({start_year}~{end_year}년) 밖이라, "
        "직선을 그대로 늘려 계산한 값이에요. 멀수록 불확실해집니다."
    )
