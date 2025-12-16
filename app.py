# dashboard built around the precomputed rapm csv outputs using streamlit

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from src.rapm_model import load_rapm_results_with_stats
from src import config


# basic page + layout stuff

st.set_page_config(
    page_title="NBA Player Impact Dashboard",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

NBA_API_SEASON = config.SEASON
HEADSHOT_BASE_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760"
HEADSHOT_PLACEHOLDER_URL = "https://placehold.co/220x160?text=No+Image"


@st.cache_data(show_spinner=False)
def load_rapm() -> pd.DataFrame: # loads the rapm resutls with stats
    df = load_rapm_results_with_stats()

    # drops any unnamed players and removes player_id 652 as they had no stats or data
    if "player" in df.columns:
        df = df[df["player"].notna()]
        df = df[df["player"].astype(str).str.strip() != ""]
    if "player_id" in df.columns:
        df = df[df["player_id"].astype(str) != "652"]

    return df


@st.cache_data(show_spinner=False)
def fetch_player_season_stats(player_id: int, season: str = NBA_API_SEASON): # fetches the player's box score stats for that season
    if player_id is None:
        return None

    try:
        from nba_api.stats.endpoints import PlayerGameLog

        game_log = PlayerGameLog(player_id=int(player_id), season=season).get_data_frames()[0]
        if game_log.empty:
            return None

        ppg = float(game_log["PTS"].mean())
        rpg = float(game_log["REB"].mean())
        apg = float(game_log["AST"].mean())
        mpg = float(game_log["MIN"].mean())

        fgm = float(game_log["FGM"].sum())
        fga = float(game_log["FGA"].sum())
        fg_pct = (fgm / fga * 100.0) if fga > 0 else None

        fg3m = float(game_log["FG3M"].sum())
        fg3a = float(game_log["FG3A"].sum())
        tp_pct = (fg3m / fg3a * 100.0) if fg3a > 0 else None

        return {
            "ppg": ppg,
            "rpg": rpg,
            "apg": apg,
            "mpg": mpg,
            "fg_pct": fg_pct,
            "tp_pct": tp_pct,
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_player_headshot_url(player_id: int) -> str: # gets the player's headshot url
    if player_id is None:
        return HEADSHOT_PLACEHOLDER_URL

    try:
        player_id_int = int(player_id)
    except (TypeError, ValueError):
        return HEADSHOT_PLACEHOLDER_URL

    url = f"{HEADSHOT_BASE_URL}/{player_id_int}.png"

    try:
        resp = requests.head(url, timeout=3)
        if resp.status_code == 200:
            return url
    except Exception:
        pass

    return HEADSHOT_PLACEHOLDER_URL


# load and pre-filter rapm data
df = load_rapm()

# the sidebar is made up of  filters for who shows up in the plots

st.sidebar.title("NBA RAPM Dashboard")
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

min_rapm = st.sidebar.slider( # slider for minimum rapm
    "Minimum RAPM",
    value=-6.0,
    min_value=-10.0,
    max_value=10.0,
    step=0.1,
)

min_stints = st.sidebar.slider( # slider for minimum stint appearances
    "Minimum Stint Appearances",
    value=0,
    min_value=0,
    max_value=int(df["stint_appearances"].max()),
)

min_minutes = st.sidebar.slider(
    "Minimum Estimated Minutes",
    value=600,
    min_value=0,
    max_value=int(df["estimated_minutes"].max()),
)

# applies filters to create df_filtered for all charts and tables
df_filtered = df[
    (df["rapm"] >= min_rapm)
    & (df["stint_appearances"] >= min_stints)
    & (df["estimated_minutes"] >= min_minutes)
].copy()

# the main header + quick summary metrics
st.title("NBA Player Impact Dashboard (RAPM)")
st.markdown("**Regularized Adjusted Plus-Minus** - Player impact ratings from the 2024-25 season")
st.caption(f"Box-score stats shown below are pulled for season: {NBA_API_SEASON}")

# summary stats for current filter state 

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns([1, 1, 2, 1])

total_players = int(len(df_filtered)) # total number of players that are shown with the current  filters
if total_players > 0:
    avg_rapm_val = float(df_filtered["rapm"].mean())
    avg_rapm_display = "0.0" if abs(avg_rapm_val) < 1e-6 else f"{avg_rapm_val:.2f}"

    top_row = df_filtered.sort_values("rapm", ascending=False).iloc[0]
    top_player_name = str(top_row["player"])
    top_rapm_display = f"{float(top_row['rapm']):.2f}"

    # when the filters are changed it defaults the player detail section to the current top player with those filters
    current_filters = (min_rapm, min_stints, min_minutes)
    prev_filters = st.session_state.get("last_filters")
    if prev_filters != current_filters:
        st.session_state["selected_player"] = top_player_name
        st.session_state["last_filters"] = current_filters
else:
    avg_rapm_display = "N/A"
    top_player_name = "N/A"
    top_rapm_display = "N/A"

with summary_col1:
    st.metric("Total Players (filtered)", total_players)
with summary_col2:
    st.metric("Avg RAPM (filtered)", avg_rapm_display)
with summary_col3:
    st.metric("Top Player (filtered)", top_player_name)
with summary_col4:
    st.metric("Top RAPM (filtered)", top_rapm_display)

st.markdown("---")

if df_filtered.empty:
    st.warning("No players match the current filters. Try relaxing the filters in the sidebar.")
else:

    # top and bottom 20 leaderboards in the form of interactive bar charts

    top20 = df_filtered.sort_values("rapm", ascending=False).head(20)
    bottom20 = df_filtered.sort_values("rapm", ascending=True).head(20)

    # using plotly bar charts so the user can hover to see orapm, drapm and minutes
    fig_top = px.bar(
        top20,
        x="rapm",
        y="player",
        orientation="h",
        hover_data=["orapm", "drapm", "estimated_minutes", "stint_appearances"],
        title="Top 20 Players by RAPM",
    )
    # shows every player name on the y axis
    fig_top.update_yaxes(
        autorange="reversed",
        tickmode="array",
        tickvals=top20["player"],
        ticktext=top20["player"],
        tickfont=dict(size=10),
    )
    fig_top.update_layout(
        height=650,
        margin=dict(l=200, r=40, t=60, b=40),
    )

    fig_bottom = px.bar( # bottom 20 players by rapm
        bottom20,
        x="rapm",
        y="player",
        orientation="h",
        hover_data=["orapm", "drapm", "estimated_minutes", "stint_appearances"],
        title="Bottom 20 Players by RAPM",
    )
    fig_bottom.update_yaxes( # shows every player name on the y axis
        autorange="reversed",
        tickmode="array",
        tickvals=bottom20["player"],
        ticktext=bottom20["player"],
        tickfont=dict(size=10),
    )
    fig_bottom.update_layout(
        height=650,
        margin=dict(l=200, r=40, t=60, b=40),
    )

    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_top, width="stretch")
        st.caption(
            "Shows the 20 highest RAPM players with the current filters. Hover to see offensive/defensive splits and usage."
        )
    
    with col2:
        st.plotly_chart(fig_bottom, width="stretch")
        st.caption(
            "Shows the 20 lowest RAPM players with the current filters. Hover to see offensive/defensive splits and usage."
        )
    

    # interactive player detail section to view individual player stats

    st.markdown("---")
    st.subheader("Player Detail")
    
    # session_state is used to keep the selected player even after filter changes
    player_options = sorted(df_filtered["player"].unique())
    default_index = 0
    if "selected_player" in st.session_state and st.session_state["selected_player"] in player_options:
        default_index = player_options.index(st.session_state["selected_player"])
    
    player_name = st.selectbox(
        "Select a player",
        options=player_options,
        index=default_index,
        key="player_select",
    )
    st.session_state["selected_player"] = player_name
    
    player_row = df_filtered[df_filtered["player"] == player_name].iloc[0]
    
    # the layout for the player detail section is theimage on the left, rapm metrics and box-score stats on the right
    img_col, info_col = st.columns([1, 2])
    
    with img_col:
        player_id = player_row.get("player_id") if "player_id" in player_row else None
        headshot_url = get_player_headshot_url(player_id)
        st.image(headshot_url, width=220)
    
    with info_col:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RAPM", f"{player_row.rapm:.2f}")
        c2.metric("ORAPM", f"{player_row.orapm:.2f}")
        c3.metric("DRAPM", f"{player_row.drapm:.2f}")
        c4.metric("Est. Minutes", f"{player_row.estimated_minutes:.0f}")
        
        # basic box score stats are fetched from the NBA API and then disolayed on the right
        stats = fetch_player_season_stats(player_id) if player_id is not None else None
        if stats is not None:
            # render box-score stats in a compact row so rapm stays the main focus
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            s1.markdown(
                f"<div style='font-size:0.9rem'><strong>PPG</strong><br>{stats['ppg']:.1f}</div>",
                unsafe_allow_html=True,
            )
            s2.markdown(
                f"<div style='font-size:0.9rem'><strong>RPG</strong><br>{stats['rpg']:.1f}</div>",
                unsafe_allow_html=True,
            )
            s3.markdown(
                f"<div style='font-size:0.9rem'><strong>APG</strong><br>{stats['apg']:.1f}</div>",
                unsafe_allow_html=True,
            )
            fg_text = f"{stats['fg_pct']:.1f}%" if stats["fg_pct"] is not None else "N/A"
            s4.markdown(
                f"<div style='font-size:0.9rem'><strong>FG%</strong><br>{fg_text}</div>",
                unsafe_allow_html=True,
            )
            tp_text = f"{stats['tp_pct']:.1f}%" if stats["tp_pct"] is not None else "N/A"
            s5.markdown(
                f"<div style='font-size:0.9rem'><strong>3P%</strong><br>{tp_text}</div>",
                unsafe_allow_html=True,
            )
            s6.markdown(
                f"<div style='font-size:0.9rem'><strong>MPG</strong><br>{stats['mpg']:.1f}</div>",
                unsafe_allow_html=True,
            )
        
        st.caption(
            "Shows the selected player's RAPM impact alongside basic box-score production for the current season."
        )
    
    # bar chart comparing this specific player's orapm and drapm side by side
    fig_player_split = px.bar(
        x=["ORAPM", "DRAPM"],
        y=[player_row.orapm, player_row.drapm],
        title=f"{player_name} ORAPM vs DRAPM",
        labels={"x": "", "y": "RAPM"},
    )
    st.plotly_chart(fig_player_split, width="stretch")
    st.caption("Compares this player's offensive and defensive RAPM contributions.")

# rapm distribution + orapm vs drapm scatter plot

st.markdown("---")
st.subheader("RAPM Distribution & ORAPM vs DRAPM")

dist_col1, dist_col2 = st.columns(2)

# creating a plotly histogram of the rapm distribution with boundaries so the user can differentiate
with dist_col1:
    fig_hist = px.histogram( # creates the histogram of the rapm distribution
        df_filtered,
        x="rapm",
        nbins=30,
        title="RAPM Distribution",
    )
    fig_hist.update_traces(
        marker_color="#2E86AB",
        marker_line_color="white",
        marker_line_width=1.0,
        opacity=0.85,
    )
    st.plotly_chart(fig_hist, width="stretch") 
    st.caption(
        "Distribution of RAPM values for all filtered players. Bars to the right mean higher overall impact."
    )

# use plotly to create a scatter plot for orapm vs drapm
with dist_col2:
    fig_scatter = px.scatter( # creates scatter plot for orapm vs drapm
        df_filtered,
        x="orapm",
        y="drapm",
        color="rapm",
        hover_name="player",
        hover_data=["estimated_minutes", "stint_appearances"],
        title="ORAPM vs DRAPM (colored by Total RAPM)",
        color_continuous_scale="RdYlGn",
    )
    st.plotly_chart(fig_scatter, width="stretch")
    st.caption(
        "Each point is a player, with the x-axis being offensive RAPM, the y-axis being defensive RAPM, and the color showing total RAPM."
    )

st.markdown("---")
st.subheader("League-wide RAPM Uncertainty vs Minutes")

# an uncertainty plot made up of ci width vs minutes
if {"ci_low", "ci_high", "estimated_minutes", "rapm", "player"}.issubset(df.columns):
    df_ci = df.copy()
    mask_ci = (
        df_ci["ci_low"].notna()
        & df_ci["ci_high"].notna()
        & df_ci["estimated_minutes"].notna()
    )
    df_ci = df_ci[mask_ci].copy() 
    if not df_ci.empty:
        df_ci["ci_width"] = df_ci["ci_high"] - df_ci["ci_low"]
        fig_uncertainty = px.scatter( # creates scatter plot for uncertainty vs estimated minutes
            df_ci,
            x="estimated_minutes",
            y="ci_width",
            color="rapm",
            hover_name="player",
            title="RAPM Uncertainty vs Estimated Minutes (League-wide)",
            labels={
                "estimated_minutes": "Estimated Minutes",
                "ci_width": "95% CI Width (Total RAPM)",
                "rapm": "Total RAPM",
            },
            color_continuous_scale="RdYlGn", # uses a color scale for the rapm values
        )
        st.plotly_chart(fig_uncertainty, width="stretch") 
        st.caption(
            "Each point is a player. The X-axis shows estimated minutes and the y-axis shows how wide the 95% CI is. "
            "Uncertainty should shrink (lower CIs) as minutes increase, but it doesn't, which is a highlights that RAPM is very dependent on sample-size. "
        )


