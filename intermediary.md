# Segmentation d'IRM du coeur: rendu intermédiaire
*par **Quentin Révillon, Benjamin Dupuis***

## Présentation du projet:
En se basant sur des techniques d'analyse d'image traditionnelle, et par opposition aux approches plus récente de Deep-Learning (type CNN), l'objectif de notre projet est de fournir une méthode de segmentation du ventricule gauche du coeur sur des images IRM. Cette méthode doit être robuste, précise et rapide.

## Plannification:
Pour mener à bien ce projet, nous avons décidé de nous concentrer sur l'algorithme de segmentation par croissance de région. Nous avons défini les étapes suivantes:
1. **Prétraitement des images**: normalisation, filtrage, etc.
2. **Identification d'un *seed point***: point initial proche du centre du ventricule.
3. **Segmentation par croissance de région**: à partir du *seed point*, on agrandit la région selon des conditions à définir.

## Bilan intermédiaire:
Pour l'instant, voici le détail des tâches que nous avons réalisées:
- **Prévisualisation et familiarisation avec les données**: nous avons visualisé les images IRM et les masques associés. Nous avons également analysé la distribution du jeu de données.