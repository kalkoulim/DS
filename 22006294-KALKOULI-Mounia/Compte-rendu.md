---

# 📘 FOREST FIRES


![FIRE](FIRE.png)

---

## 1. Le Contexte Métier et la Mission

### Le Problème
Les feux de forêt causent des pertes économiques et écologiques majeures, et menacent directement les populations et les infrastructures.​
Objectif : prédire la surface brûlée d’un feu de forêt dans le parc de Montesinho (Portugal) à partir de données météo et d’indices de sécheresse.

​

Enjeu métier :

Anticiper la gravité d’un incendie pour adapter les moyens de prévention et de lutte (alerte, mobilisation des équipes, évacuation).

La “mauvaise” erreur n’est pas symétrique :

Sous‑estimer une grande surface brûlée (prédire petit alors que le feu sera grand) → moyens insuffisants, dégâts majeurs.

Sur‑estimer une surface (prédire grand pour un petit feu) → surcoût opérationnel, mais risque humain plus faible.
        Dans ce contexte, on cherchera à mieux prédire les grands feux et/ou à réduire fortement les grosses sous‑estimations (métriques de type RMSE, quantiles de l’erreur, courbes REC comme dans Cortez & Morais)

### Les Données 
On utilise le dataset Forest Fires de l’UCI Machine Learning Repository, 517 feux, 12 features + 1 cible area.

​

X (features) :

Spatiales : X, Y (coordonnées sur la carte du parc, 1–9).

​

Temporelles : month (jan–dec), day (mon–sun).

​

Indices FWI : FFMC, DMC, DC, ISI (indices de sécheresse / inflammabilité).

​

Météo directe : temp (°C), RH (% humidité), wind (km/h), rain (mm/m²).
​y (target) :

area : surface brûlée en hectares, de 0 à ~1090 ha, très fortement concentrée près de 0 (beaucoup de petits feux).

​

Dans l’article original, ln(area + 1) est utilisé pour rendre le problème de régression plus stable.

---

## 2. Le Code Python (Laboratoire)

Ce script est votre paillasse de laboratoire. Il contient toutes les manipulations nécessaires.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings("ignore")

# --- PHASE 1 : ACQUISITION ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
df = pd.read_csv(url)

# --- PHASE 2 : DATA WRANGLING ---
# Pas de valeurs manquantes selon la description UCI
# Transformation de la cible (comme dans Cortez & Morais 2007)
df["area_log"] = np.log(df["area"] + 1)

X = df.drop(columns=["area", "area_log"])
y = df["area_log"]

num_cols = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
cat_cols = ["month", "day"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(drop="first"), cat_cols),
    ]
)

# --- PHASE 3 : EDA légère ---
print("--- Aperçu ---")
print(df.head())
print("\n--- Statistiques area ---")
print(df["area"].describe())

plt.figure(figsize=(6, 4))
sns.histplot(df["area"], bins=50)
plt.title("Distribution de la surface brûlée (ha)")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df["area_log"], bins=50)
plt.title("Distribution de ln(area + 1)")
plt.show()

# --- PHASE 4 : PROTOCOLE EXPÉRIMENTAL ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- PHASE 5 : MODELE (Random Forest) ---
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", model)
])

pipe.fit(X_train, y_train)

# --- PHASE 6 : EVALUATION ---
y_pred_log = pipe.predict(X_test)
# Retour à l’échelle originale
y_test_area = np.expm1(y_test)
y_pred_area = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test_area, y_pred_area)
rmse = mean_squared_error(y_test_area, y_pred_area, squared=False)

print(f"\nMAE (ha) : {mae:.2f}")
print(f"RMSE (ha) : {rmse:.2f}")

plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test_area, y=y_pred_area)
plt.plot([0, max(y_test_area)], [0, max(y_test_area)], "r--")
plt.xlabel("Surface réelle (ha)")
plt.ylabel("Surface prédite (ha)")
plt.title("Réelle vs prédite (échelle originale)")
plt.show()

 
```

---

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

Valeurs manquantes et qualité des données

La documentation UCI indique aucune valeur manquante sur ce dataset.

​

En pratique, on vérifie quand même (df.isna().sum()) et la présence de quelques doublons possibles.

    ​

Cible transformée : ln(area + 1)

area est ultra‑skewée : la plupart des feux brûlent moins de 1 ha, quelques cas extrêmes dépassent 500 ha.

​

La transformation ln⁡(area+1)ln(area+1) :

Compresse les gros feux (réduit le poids des extrêmes).

Rapproche la distribution d’une forme plus “gaussienne”, ce qui stabilise de nombreux modèles.

    ​

Comme dans le guide médical, il faut penser à l’échelle de la cible : ici, les métriques finales doivent être interprétées en hectares
(d’où la re‑transformation avec expm1).

    ​

Encodage des variables catégorielles

month et day sont nominales (pas ordinales strictement dans cette formulation UCI), on utilise donc One‑Hot Encoding.

​

Attention au data leakage : l’encodeur est appris dans le Pipeline, donc uniquement sur le train, puis appliqué au test, ce qui évite de “voir le futur”. (Même principe que pour l’imputation dans ton guide initial, mais appliqué à l’encodage.)


---

## 4. Analyse Approfondie : Exploration (EDA)

Distribution & skewness

Histogrammes de area et area_log :

area → massivement concentrée sur 0 avec quelques valeurs énormes.

​

area_log → plus “lisse”, plus exploitable par des modèles linéaires ou des métriques classiques.

        ​

Relations avec les features

Quelques axes d’exploration typiques :

​
Saison / mois :

Plus de feux en été (juil–sep), lié à temp élevée, RH faible, DC et ISI élevés.

Météo directe :

temp : les feux importants sont plus probables à températures élevées.

rain : souvent 0 au moment de l’incident, les grandes surfaces brûlées surviennent en absence de pluie.

Indices FWI :

DC (sécheresse à long terme) et ISI (vitesse de propagation) ont tendance à être plus élevés pour les feux plus importants.

---

## 5. Analyse Approfondie : Méthodologie (Split)

Objectif : généralisation vs surapprentissage

On cherche un modèle qui donne une bonne précision moyenne, mais surtout qui ne sous‑estime pas de façon catastrophique certains grands feux.

​

Split classique : train_test_split(test_size=0.2, random_state=42) (80/20).

Possibilités d’aller plus loin :

k‑fold cross‑validation (ex : 10 folds) pour stabiliser les mesures étant donné la petite taille du dataset (517 lignes).

​

Répéter les splits (comme Cortez & Morais : 10‑fold × 30 runs) pour mieux évaluer la robustesse du modèle

---

## 6. FOCUS THÉORIQUE : L'Algorithme Random Forest 🌲

La logique générale est la même que dans ton exemple médical, mais appliquée à une cible continue.

​

Chaque arbre de décision apprend une fonction “if/else” qui prédît une surface brûlée à partir des features (par exemple, si DC > seuil et ISI > seuil alors feu plus grand).

Le bagging + aléa sur les features :

Bootstrap sur les lignes → arbres variés.

Sous‑ensemble aléatoire de variables considérées à chaque split → explore différentes combinaisons météo/spatiales.

        ​

En sortie, pour un nouvel incident :

Chaque arbre donne une prédiction numérique (surface log‑transformée).

La Random Forest moyenne ces valeurs pour donner la prédiction finale (puis on applique expm1).

Sur ce dataset, des travaux montrent que RF est compétitif mais que d’autres modèles (SVM gaussien sur ln(area+1) par exemple) peuvent mieux capturer les petits feux, qui sont majoritaires
---

## 7. Analyse Approfondie : Évaluation (L'Heure de Vérité)

Pour un problème de régression, la “matrice de confusion” n’existe pas, mais la logique métier reste la même : punir plus sévèrement les grosses erreurs sur les grands feux.

​
Métriques de base

MAE (Mean Absolute Error) en hectares → erreur moyenne absolue sur la surface brûlée.

RMSE (Root Mean Squared Error) en hectares → pénalise davantage les grandes erreurs (sous‑estimations ou sur‑estimations massives).

Métriques plus fines (dans l’esprit de Cortez & Morais)

REC curve (Regression Error Characteristic) : pour un seuil d’erreur E donné (par ex. 10 ha), on mesure la proportion de feux prédits avec une erreur ≤ E.

​

Permet de dire : “Dans X% des cas, l’erreur est inférieure à 10 ha.”

Analyse séparée des petits feux (area < 1 ha) vs grands feux (area > 50 ha, seuil à définir) pour vérifier que le modèle n’ignore pas les cas rares mais critiques

### Conclusion du Projet
Le projet complet consiste donc à :

Comprendre l’enjeu : prioriser la bonne allocation des moyens de lutte, donc limiter les grosses sous‑estimations des grands feux.

Construire une pipeline propre (encodage, transformation log, modèle, cross‑validation).

Choisir des métriques adaptées (MAE/RMSE sur l’échelle ha, REC, analyse des grands feux) et non se limiter à un score unique.

C’est la même “anatomie” de projet que dans ton exemple médical, mais transposée à un problème de régression environnementale plutôt qu’à un problème de classification médicale.
