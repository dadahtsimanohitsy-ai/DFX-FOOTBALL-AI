import math
from datetime import date
import pandas as pd
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="DFX FOOTBALL AI", page_icon="⚽", layout="wide")

st.markdown("""
<style>
.stApp {background: #071810; color: #eef7ef;}
[data-testid="stMetric"] {background:#10291d; padding:14px; border-radius:12px;}
div[data-testid="stDataFrame"] {border:1px solid #244b35;}
</style>
""", unsafe_allow_html=True)

st.title("⚽ DFX FOOTBALL AI")
st.caption("DATA • ANALYSIS • PREDICTION | Prototype v1.0 — résultats historiques uniquement")

REQUIRED = ["league", "date", "home_team", "away_team", "home_goals", "away_goals"]

def normalize(df):
    df = df.copy()
    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante : {c}")
    df["league"] = df["league"].astype(str).str.strip()
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    df = df[(df.home_goals >= 0) & (df.away_goals >= 0)]
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=REQUIRED)

with st.sidebar:
    st.header("Data Center")
    uploaded = st.file_uploader("Importer un historique CSV", type=["csv"])
    if uploaded is not None:
        try:
            incoming = normalize(pd.read_csv(uploaded))
            st.session_state.history = pd.concat(
                [st.session_state.history, incoming], ignore_index=True
            ).drop_duplicates(subset=["league","date","home_team","away_team","home_goals","away_goals"])
            st.success(f"{len(incoming)} ligne(s) importée(s).")
        except Exception as e:
            st.error(f"Import impossible : {e}")

    st.download_button(
        "Télécharger l'historique CSV",
        data=st.session_state.history.to_csv(index=False).encode("utf-8"),
        file_name="dfx_historique.csv",
        mime="text/csv"
    )

tab_pred, tab_add, tab_hist, tab_guide = st.tabs(
    ["🔮 Prédictions", "➕ Ajouter un résultat", "📚 Historique", "ℹ️ Guide"]
)

with tab_add:
    st.subheader("Saisie manuelle d'un résultat terminé")
    with st.form("add_result"):
        c1, c2 = st.columns(2)
        league = c1.selectbox("Ligue", ["English League", "German League"])
        match_date = c2.date_input("Date du match", value=date.today())
        home = c1.text_input("Équipe domicile")
        away = c2.text_input("Équipe extérieur")
        hg = c1.number_input("Buts domicile", min_value=0, max_value=30, value=1)
        ag = c2.number_input("Buts extérieur", min_value=0, max_value=30, value=1)
        submitted = st.form_submit_button("Enregistrer le résultat")
        if submitted:
            if not home.strip() or not away.strip() or home.strip().lower() == away.strip().lower():
                st.error("Entre deux noms d'équipes différents.")
            else:
                row = pd.DataFrame([{
                    "league": league, "date": pd.Timestamp(match_date),
                    "home_team": home.strip(), "away_team": away.strip(),
                    "home_goals": int(hg), "away_goals": int(ag)
                }])
                st.session_state.history = pd.concat(
                    [st.session_state.history, row], ignore_index=True
                )
                st.success("Résultat ajouté à la session. Télécharge le CSV pour le conserver.")

with tab_pred:
    st.subheader("Analyse d'un prochain match")
    hist = st.session_state.history
    if hist.empty:
        st.warning("Commence par importer un CSV ou ajouter des résultats terminés. Aucun pronostic fiable sans historique.")
    else:
        leagues = sorted(hist.league.unique().tolist())
        selected_league = st.selectbox("Compétition", leagues)
        league_df = hist[hist.league == selected_league].copy()
        teams = sorted(set(league_df.home_team).union(league_df.away_team))
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Équipe domicile", teams, key="pred_home")
        away_options = [t for t in teams if t != home_team]
        away_team = col2.selectbox("Équipe extérieure", away_options, key="pred_away") if away_options else None
        if away_team:
            max_goals = st.slider("Limite de buts par équipe pour la matrice de scores", 4, 10, 7)
            if st.button("Calculer les probabilités", type="primary"):
                # Each team's attack rate is blended with the opponent's defensive concession rate.
                league_home_avg = max(0.05, league_df.home_goals.mean())
                league_away_avg = max(0.05, league_df.away_goals.mean())
                league_total = max(0.1, (league_df.home_goals.mean() + league_df.away_goals.mean()) / 2)

                def team_stats(team):
                    home_rows = league_df[league_df.home_team == team]
                    away_rows = league_df[league_df.away_team == team]
                    scored = pd.concat([home_rows.home_goals, away_rows.away_goals])
                    conceded = pd.concat([home_rows.away_goals, away_rows.home_goals])
                    return scored, conceded

                hs, hc = team_stats(home_team)
                aws, awc = team_stats(away_team)
                # Conservative fallback to league averages for teams with sparse history.
                home_attack = (hs.mean() if len(hs) else league_home_avg)
                home_defence = (hc.mean() if len(hc) else league_away_avg)
                away_attack = (aws.mean() if len(aws) else league_away_avg)
                away_defence = (awc.mean() if len(awc) else league_home_avg)
                lam_home = max(0.05, (home_attack + away_defence) / 2)
                lam_away = max(0.05, (away_attack + home_defence) / 2)

                # Independent Poisson approximation: exploratory baseline, not a proven virtual-football edge.
                goals = list(range(max_goals + 1))
                p_home = [poisson.pmf(k, lam_home) for k in goals]
                p_away = [poisson.pmf(k, lam_away) for k in goals]
                matrix = [[p_home[i] * p_away[j] for j in goals] for i in goals]
                p1 = sum(matrix[i][j] for i in goals for j in goals if i > j)
                px = sum(matrix[i][j] for i in goals for j in goals if i == j)
                p2 = sum(matrix[i][j] for i in goals for j in goals if i < j)
                total_mass = sum(sum(r) for r in matrix)
                # Normalize the truncated score matrix.
                p1, px, p2 = [v / total_mass for v in (p1, px, p2)]
                score_rows = []
                for i in goals:
                    for j in goals:
                        score_rows.append((f"{i}-{j}", matrix[i][j] / total_mass))
                score_rows.sort(key=lambda x: x[1], reverse=True)
                top_scores = score_rows[:5]

                def over_prob(line):
                    prob = sum(matrix[i][j] for i in goals for j in goals if i+j > line)
                    return prob / total_mass

                matches_home = len(hs)
                matches_away = len(aws)
                sample = min(matches_home, matches_away)
                confidence = "Faible"
                if sample >= 20:
                    confidence = "Moyenne — modèle simple"
                if sample >= 50:
                    confidence = "À tester davantage — pas une garantie"

                st.markdown("### Résultat 1X2")
                m1, m2, m3 = st.columns(3)
                m1.metric("1 — Domicile", f"{p1*100:.1f}%")
                m2.metric("N — Nul", f"{px*100:.1f}%")
                m3.metric("2 — Extérieur", f"{p2*100:.1f}%")
                st.markdown("### Over / Under")
                ou_rows = []
                for line in [0.5, 1.5, 2.5, 3.5]:
                    over = over_prob(line)
                    ou_rows.append({"Ligne": f"{line:.1f}", "Over": f"{over*100:.1f}%",
                                    "Under": f"{(1-over)*100:.1f}%"})
                st.dataframe(pd.DataFrame(ou_rows), use_container_width=True, hide_index=True)
                st.markdown("### Scores exacts les plus probables")
                st.dataframe(pd.DataFrame(
                    [{"Score": s, "Probabilité": f"{p*100:.2f}%"} for s,p in top_scores]
                ), use_container_width=True, hide_index=True)
                st.info(
                    f"Échantillon : {matches_home} matchs pour {home_team}, "
                    f"{matches_away} pour {away_team}. Niveau indicatif : {confidence}. "
                    f"Buts moyens estimés : {lam_home:.2f} – {lam_away:.2f}."
                )
                st.caption("Les probabilités proviennent d'un modèle de Poisson indépendant sur les données importées. "
                           "Les scores non inclus dans la matrice sont ignorés puis les probabilités restantes sont normalisées.")

with tab_hist:
    st.subheader("Historique enregistré dans cette session")
    if st.session_state.history.empty:
        st.info("Aucun résultat pour le moment.")
    else:
        st.write(f"Total : {len(st.session_state.history)} résultat(s)")
        st.dataframe(st.session_state.history.sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)
        if st.button("Vider l'historique de la session"):
            st.session_state.history = pd.DataFrame(columns=REQUIRED)
            st.rerun()

with tab_guide:
    st.subheader("Comment utiliser la version 1.0")
    st.markdown("""
1. Récupère uniquement les résultats auxquels tu as légalement accès.
2. Prépare un fichier CSV avec les colonnes : `league,date,home_team,away_team,home_goals,away_goals`.
3. Utilise exactement les mêmes noms d'équipes à chaque ligne.
4. Importe le CSV dans le panneau latéral ou ajoute les scores manuellement.
5. Sélectionne la ligue et les deux équipes, puis calcule les probabilités.
6. Télécharge régulièrement ton historique CSV : la mémoire de cette version est temporaire.

**Limites importantes**
- Cette version ne se connecte pas à Bet261 et ne récupère pas encore automatiquement ses résultats.
- Le modèle Poisson est une base statistique exploratoire. Il ne prédit pas un RNG et ne garantit aucun gain.
- Pour le football virtuel, il faut d'abord vérifier le fournisseur, le mécanisme des résultats et la disponibilité d'un flux officiel autorisé.
- Teste le modèle sur des matchs passés qui n'ont pas servi à le construire avant d'évaluer sa qualité.
""")
