Le dataset “Autistic Spectrum Disorder Screening Data for Children” (Thabtah, 2017)
<img src="DS & ML/caleb-woods-ecRuhwPIW7c-unsplash.jpg" style="height:1000px;margin-right:1000px"/> 


--


Le dataset “Autistic Spectrum Disorder Screening Data for Children” (Thabtah, 2017) regroupe des données issues de questionnaires de dépistage de l’autisme chez les enfants. Chaque observation correspond à un enfant évalué selon plusieurs critères comportementaux, démographiques et médicaux. Le but de l’étude est de déterminer si un enfant présente un risque d'autisme à partir des réponses à un ensemble d'indicateurs.

### Problématqiue:
La problématique principale est donc la suivante :
peut-on construire un modèle d’apprentissage supervisé capable de prédire, à partir des caractéristiques d’un enfant, si celui-ci est susceptible de présenter un trouble du spectre autistique (ASD)??

### La nature de la variable:
La nature de la variable cible fournie dans la base de données — généralement nommée Class/ASD ou ASD_trait — indique si le dépistage est positif (1) ou négatif (0). Cela signifie que la tâche à résoudre est un problème de classification supervisée binaire, dont l’objectif est de distinguer deux catégories : 
    0 : enfant ne présentant pas de signes autistiques
    1 : enfant présentant des signes autistiques

Donc la problématique consiste à prédire si un enfant présente un risque d’autisme à partir des variables de dépistage (score AQ, comportement, âge, etc.).
C’est un problème de classification supervisée binaire, où l’objectif est de construire un modèle capable de distinguer deux classes : ASD vs non-ASD.

## Taille et structure globale
  Nombre d’instances : 292. 
  Nombre de variables (features) : 20 (hors éventuellement la variable cible), ou 21 si l’on inclut la target — selon la version/documentation utilisée. 
  Présence de valeurs manquantes (« Has Missing Values? Yes »). 
  Type global : « Multivariate » — c.-à-d. plusieurs variables explicatives. 
  Tâche visée : classification (supervisée). 

## Types de variables et signification des features

Le dataset combine :
  Des features issues d’un questionnaire de dépistage comportemental — typiquement 10 questions, codées en entier (souvent binaire). 
  Des caractéristiques démographiques / contextuelles : âge, genre, ethnie, antécédents (jaunisse à la naissance), histoire familiale, etc. 

## Identification de la target (variable cible)

- La variable cible (target) indique si l’enfant est classé “ASD” (risque / dépistage positif) ou non. 
- Le label est typiquement binaire : par exemple “1” = ASD, “0” = non-ASD. 
- Le score global “screening score” (souvent calculé à partir des réponses A1–A10) peut aussi être présent dans certaines versions du dataset — mais ce score est souvent utilisé pour déterminer la valeur de la target (c.-à-d. si le score dépasse un seuil → ASD). 

