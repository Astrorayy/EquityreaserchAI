# Equity Research AI

Application locale de recherche actions (Python + Streamlit) pour analyser et valoriser
**n'importe quelle société cotée**. Deux profils sont préconfigurés et vérifiés :
**Managem** (Bourse de Casablanca) et **SanDisk Corporation** (Nasdaq : SNDK).
Interface en français, code et documentation en anglais.

### Ajouter une société

Dans la barre latérale :

- **🔎 Rechercher une autre société** — par nom ou ticker. Deux sources sans clé :
  - la **table officielle SEC** (toutes les sociétés enregistrées aux États-Unis) : le
    CIK est récupéré, ce qui permet l'**import automatique des comptes publiés**
    (page États financiers → SEC EDGAR) ;
  - **Yahoo Finance** (non officiel) pour les autres places (Paris, Francfort, Londres,
    Toronto…) : identité, devise et place sont **à vérifier** et signalées comme telles ;
    les comptes s'importent ensuite par PDF ou saisie.
- **✍️ Ajouter une société manuellement** — pour une place non couverte (ex. Bourse de
  Casablanca) : nom, ticker, place, pays, devise, secteur, clôture d'exercice.

Une société ajoutée peut être retirée depuis la page **Recherche société** (ses données
sont conservées et réapparaissent si elle est ajoutée à nouveau).

L'application fonctionne **sans clé API et sans connexion Internet** : les comptes peuvent
être saisis manuellement ou chargés depuis un jeu de démonstration explicite. Les
intégrations externes (SEC EDGAR, sites investisseurs, recherche web, LLM) sont
optionnelles et activées par variables d'environnement.

> Outil de recherche et de formation. **Ce n'est pas un conseil en investissement** et
> aucune donnée n'est inventée : chaque chiffre est tracé jusqu'à sa source ou signalé
> comme manquant.

---

## 1. Fonctionnalités

| Page | Contenu |
|---|---|
| **Tableau de bord** | Vue d'ensemble de la société sélectionnée, KPI, alertes de qualité, cours de référence. |
| **Recherche société** | Identité, périmètre de consolidation, documents officiels (SEC EDGAR / site investisseurs), sources secondaires. |
| **États financiers** | Comptes annuels et trimestriels, statut de chaque poste (publié / calculé / estimé / manuel / simulé), import PDF/HTML, saisie manuelle. |
| **Valorisation** | DCF FCFF : WACC, croissance perpétuelle, scénarios baissier / central / haussier, table de sensibilité WACC × g, tornade, pont VE → fonds propres. |
| **Comparables** | Multiples de pairs (VE/EBITDA, P/E, P/B), médianes, valeurs implicites, exclusions motivées (devise, EBITDA négatif, données manquantes). |
| **Assistant IA** | Optionnel. Chat qui appelle des outils Python déterministes ; refuse d'inventer, sépare faits / calculs / hypothèses. Désactivé sans clé API. |
| **Sources et qualité** | Journal des sources, horodatages de récupération, contrôles comptables et avertissements. |
| **Paramètres** | Langue, chemins de données, état des fournisseurs, configuration LLM / recherche. |

Rapports : export **HTML** autonome et **PDF** (fpdf2) depuis la page Valorisation.

---

## 2. Démarrage rapide (Windows)

Prérequis : Python **3.11 ou 3.12** (64 bits), ~250 Mo d'espace disque, pas de GPU.

```powershell
cd C:\Users\azuz\Desktop\equityresearchAI

# 1. Environnement virtuel
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. (Optionnel) yfinance + lxml — données de marché indicatives pour les tickers US
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt

# 3. (Optionnel) configuration : copier le modèle puis renseigner les clés
Copy-Item .env.example .env

# 4. Lancer l'application
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Le navigateur s'ouvre sur `http://localhost:8501`. Sans `.env`, l'application démarre
normalement : l'assistant IA est simplement désactivé.

> Sur Windows, `python` peut pointer vers le stub du Microsoft Store. Utilisez toujours
> le chemin explicite `.\.venv\Scripts\python.exe` (ou `py -3.12`).

### Données de démonstration

Dans **Paramètres → Jeu de démonstration**, le bouton « Charger le jeu de démonstration
simulé » attache un jeu **SIMULÉ** à la société sélectionnée. Il est étiqueté comme tel
dans toute l'interface et dans les rapports. Il ne remplace jamais des chiffres publiés.

### Données réelles

- **SanDisk (SNDK)** : formulaire 10-K/10-Q et XBRL via l'API SEC EDGAR (aucune clé
  requise, un `User-Agent` descriptif est recommandé — voir `ERAI_SEC_USER_AGENT`).
- **Managem** : rapports annuels et semestriels depuis le site investisseurs de Managem,
  bulletin de la Bourse de Casablanca, publications AMMC. Les documents peuvent aussi
  être importés manuellement (PDF/HTML) puis vérifiés poste par poste.
- **Cours** : saisie manuelle pour Casablanca ; yfinance (non officiel) pour les tickers
  américains, si installé.

---

## 3. Configuration (`.env`)

Toutes les variables sont **optionnelles**. Voir `.env.example` pour la liste complète.

| Variable | Défaut | Rôle |
|---|---|---|
| `ERAI_LANG` | `fr` | Langue de l'interface (`fr` / `en`). |
| `ERAI_DATA_DIR` | `./data` | Base SQLite, cache HTTP et rapports. |
| `ERAI_HTTP_TIMEOUT` | `20` | Délai HTTP (secondes). |
| `ERAI_SEC_USER_AGENT` | prototype | `User-Agent` SEC avec e-mail de contact. |
| `ERAI_SEARCH_PROVIDER` | `none` | `duckduckgo` (sans clé), ou `tavily` / `brave` / `serper` (clé requise). |
| `ERAI_SEARCH_API_KEY` | — | Clé du fournisseur de recherche (sauf DuckDuckGo). |
| `ERAI_LLM_BASE_URL` | OpenAI | Tout point d'accès compatible *chat completions*. |
| `ERAI_LLM_API_KEY` | — | **Vide = assistant IA désactivé.** |
| `ERAI_LLM_MODEL` | `gpt-4o-mini` | Nom du modèle chez le fournisseur. |
| `ERAI_LLM_TEMPERATURE` | `0.2` | Température d'échantillonnage. |
| `ERAI_ENABLE_YFINANCE` | `true` | Désactiver les cours automatiques US. |

Exemples de fournisseurs à petit budget / offre gratuite : Groq, Google Gemini,
OpenRouter, Mistral, ou Ollama en local (`http://localhost:11434/v1`).
L'interface LLM est indépendante du fournisseur (API compatible OpenAI).

### Activer la recherche web

La recherche web est **optionnelle** et désactivée par défaut. Pour l'activer :

- **Sans clé API (le plus simple)** : `ERAI_SEARCH_PROVIDER=duckduckgo`.
  L'application utilise le paquet `ddgs` (installé par `requirements.txt`), qui
  interroge plusieurs moteurs publics. Ces moteurs limitent parfois le débit :
  un échec temporaire est signalé, il suffit de réessayer.
- **Avec clé API (plus stable)** : `ERAI_SEARCH_PROVIDER=tavily` (offre gratuite),
  `brave` ou `serper`, et renseignez `ERAI_SEARCH_API_KEY`.

Les résultats de recherche sont toujours présentés comme des **pistes** avec leur
URL, jamais comme des chiffres officiels : ils doivent être vérifiés dans le
document source avant toute utilisation.

### Déploiement sur Streamlit Community Cloud

Le dépôt ne contient **pas** `.env` (volontairement ignoré). Sur Streamlit Cloud, la
configuration se fait dans **Manage app → Settings → Secrets**, au format TOML :

```toml
ERAI_SEARCH_PROVIDER = "duckduckgo"

# Optionnel — assistant IA
# ERAI_LLM_BASE_URL = "https://api.groq.com/openai/v1"
# ERAI_LLM_API_KEY = "..."
# ERAI_LLM_MODEL = "..."
```

Ces secrets sont lus au démarrage et exposés comme variables d'environnement
(`config.apply_streamlit_secrets`), sans jamais écraser `.env` ni l'environnement local.

Après un `git push`, Streamlit Cloud reconstruit normalement l'application. Si ce n'est
pas le cas : **Manage app → ⋮ → Reboot app**, et vérifiez que la branche déployée est
`main` avec `app.py` comme fichier principal. Le nouvel onglet **Sources → Recherche web**
et la ligne « Recherche web » de la page **Paramètres** confirment que le code déployé
est à jour.

> La base SQLite et les rapports vivent dans le système de fichiers éphémère de
> Streamlit Cloud : ils sont réinitialisés à chaque redémarrage. Exportez les rapports
> ou utilisez un stockage externe si vous devez conserver les données.

---

## 4. Méthodologie et garde-fous

### DCF (FCFF)

- Flux de trésorerie disponibles pour l'entreprise : `NOPAT + amortissements − capex − Δ BFR`.
- Actualisation au **WACC** ; valeur terminale par croissance perpétuelle de Gordon.
- Contrainte stricte **WACC > g** (erreur explicite ; les cellules invalides des tables de
  sensibilité affichent `n.d.` au lieu d'être extrapolées).
- Scénarios baissier / central / haussier et table de sensibilité WACC × croissance.
- Part de la valeur terminale dans la valeur d'entreprise affichée et signalée si > 85 %.
- Passage VE → fonds propres : `VE − dette nette − autres créances seniors + actifs hors exploitation`.
  La dette nette n'est **jamais** soustraite deux fois ; un DCF FCFF n'utilise jamais le
  coût des fonds propres comme taux d'actualisation.
- Les valeurs sont en **millions** de la devise de reporting ; les actions en **millions**.
  Une devise non convertie est signalée, jamais additionnée silencieusement.

### Multiples

- Médiane des pairs (moins sensible que la moyenne aux valeurs extrêmes).
- Pairs d'une autre devise exclus par défaut (un multiple est sans dimension, mais les
  agrégats monétaires ne sont pas comparables sans taux de change déclaré).
- Exclusions motivées : EBITDA ou résultat net négatif, valeur d'entreprise manquante,
  division par zéro.

### Modules sectoriels

- **Mines (Managem)** : marge normalisée, sensibilité aux prix des métaux, VAN de projet
  minier, risques pays / devises / ressources.
- **Semi-conducteurs (SanDisk)** : marges de **milieu de cycle** (NAND), volatilité des prix,
  concentration client, intensité capitalistique.

### Provenance et qualité

Chaque observation conserve : `source_id`, URL, date de publication, horodatage de
récupération, période, devise et unité, méthode d'extraction, statut
(`reported` / `derived` / `estimated` / `manual` / `simulated`) et notes.
Les faits publiés, les calculs internes et les hypothèses d'analyste sont affichés
séparément. Les contrôles comptables (actif = passif, somme des trimestres, cohérence des
signes, devises) produisent des avertissements explicites.

---

## 5. Structure du projet

```
app.py                     # point d'entrée Streamlit, 8 pages
workflows.py               # orchestration partagée UI / outils IA
config.py                  # réglages typés, lecture de .env
i18n.py                    # libellés FR / EN
financial_engine/          # schémas, validation, DCF, ratios, sensibilité,
                           # multiples, modèles sectoriels
data_sources/              # HTTP, SEC EDGAR, sources officielles, PDF/HTML,
                           # recherche web, cours, jeu simulé
storage/                   # SQLite (schéma) + repository (CRUD)
dashboard/                 # composants et graphiques Plotly
reports/                   # génération HTML et PDF
agent/                     # fournisseur LLM, outils, prompts, boucle d'agent
tests/                     # pytest (moteurs, stockage, UI smoke opt-in)
data/                      # base SQLite, cache, rapports générés (ignoré par git)
```

Direction des dépendances : `agent → workflows → financial_engine`.
L'interface et l'assistant produisent donc exactement les mêmes chiffres.

---

## 6. Tests

```powershell
# Suite standard (rapide, sans réseau, sans Streamlit)
.\.venv\Scripts\python.exe -m pytest tests/ -q

# Smoke test complet de l'interface (8 pages, ~15-60 s)
$env:ERAI_RUN_UI_SMOKE = '1'
$env:ERAI_DATA_DIR = './_smoke_data'
.\.venv\Scripts\python.exe -m pytest tests/test_ui_smoke.py -q
```

Couverture : DCF (résultats de référence, WACC ≤ g, séries invalides, flux négatifs,
part terminale, réconciliation VE → fonds propres), ratios (divisions par zéro, données
manquantes, dénominateurs négatifs), validation comptable, sensibilité, multiples,
modèles sectoriels, stockage SQLite (upserts, doublons, rechargement, pairs, sources),
et rendu des 8 pages via `streamlit.testing.v1.AppTest`.

---

## 7. Limites connues

- **Bourse de Casablanca** : la chaîne TLS du site échoue parfois localement
  (`CERTIFICATE_VERIFY_FAILED`). L'application le signale et invite à saisir le cours
  manuellement ou à importer le bulletin officiel ; elle ne désactive **jamais** la
  vérification des certificats.
- **yfinance** est une source non officielle, indicative, réservée aux tickers US.
- Les données ne sont pas mises à jour en temps réel ; chaque écran affiche l'horodatage
  de récupération et la période de reporting.
- Les chiffres extraits automatiquement doivent être vérifiés contre le document officiel ;
  l'interface le rappelle et conserve le statut de vérification.
- L'assistant IA peut se tromper ou mal interpréter un document : il cite ses sources, et
  n'exécute jamais de code renvoyé par le modèle. Sans clé API, il est désactivé.
- Aucun conseil en investissement, aucune recommandation d'achat ou de vente.

---

## 8. Dépannage

| Symptôme | Cause probable / solution |
|---|---|
| `python` ouvre le Microsoft Store | Utiliser `.\.venv\Scripts\python.exe -m ...`. |
| L'assistant affiche « désactivé » | Renseigner `ERAI_LLM_API_KEY` dans `.env`, puis redémarrer Streamlit. |
| Erreur TLS sur la Bourse de Casablanca | Connu ; saisir le cours manuellement. Ne pas désactiver la vérification TLS. |
| Page vide sur les comparables | Vérifier la devise des pairs et la présence de la capitalisation boursière. |
| Base à réinitialiser | Arrêter Streamlit, supprimer `data/equity_research.db`, relancer. |

---

*Prototype de recherche — utilisation interne. Les marques et noms de sociétés
appartiennent à leurs détenteurs respectifs.*
