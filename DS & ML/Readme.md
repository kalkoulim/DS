Le dataset “Autistic Spectrum Disorder Screening Data for Children” (Thabtah, 2017)
<img src="DS & ML/caleb-woods-ecRuhwPIW7c-unsplash.jpg" style="height:1000px;margin-right:1000px"/> 


--


Le dataset “Autistic Spectrum Disorder Screening Data for Children” (Thabtah, 2017) regroupe des données issues de questionnaires de dépistage de l’autisme chez les enfants. Chaque observation correspond à un enfant évalué selon plusieurs critères comportementaux, démographiques et médicaux. Le but de l’étude est de déterminer si un enfant présente un risque d'autisme à partir des réponses à un ensemble d'indicateurs.

### Problématqiue:
La problématique principale est donc la suivante :
peut-on construire un modèle d’apprentissage supervisé capable de prédire, à partir des caractéristiques d’un enfant, si celui-ci est susceptible de présenter un trouble du spectre autistique (ASD)??

La nature de la variable cible fournie dans la base de données — généralement nommée Class/ASD ou ASD_trait — indique si le dépistage est positif (1) ou négatif (0). Cela signifie que la tâche à résoudre est un problème de classification supervisée binaire, dont l’objectif est de distinguer deux catégories :

0 : enfant ne présentant pas de signes autistiques
1 : enfant présentant des signes autistiques

Donc la problématique consiste à prédire si un enfant présente un risque d’autisme à partir des variables de dépistage (score AQ, comportement, âge, etc.).
C’est un problème de classification supervisée binaire, où l’objectif est de construire un modèle capable de distinguer deux classes : ASD vs non-ASD.


