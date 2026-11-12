# DFX FOOTBALL AI — Prototype 1.0

## Lancer sur Windows
1. Installe Python 3.11 ou plus récent depuis https://www.python.org/downloads/
   - Pendant l'installation, coche **Add Python to PATH**.
2. Décompresse le dossier.
3. Ouvre l'invite de commandes dans ce dossier.
4. Exécute :

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

5. Le navigateur ouvre l'application localement.

## Format CSV requis
Colonnes obligatoires :
`league,date,home_team,away_team,home_goals,away_goals`

Exemple :
`English League,2026-11-10,Manchester Blue,London Reds,1,2`

## Important
La version 1.0 utilise uniquement des données importées ou saisies. Elle ne se connecte pas à Bet261.
Ne contournez pas les protections du site et n'utilisez pas de scraping nécessitant une connexion privée.
L'accès automatique sera ajouté uniquement si Bet261 ou son fournisseur propose une API/flux autorisé.
Les résultats virtuels peuvent être générés par RNG : les historiques ne garantissent pas la prédiction du prochain match.
