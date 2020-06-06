Ce projet est réalisé dans le cadre de la formation de cycle d'ingénieur (première année) de l'Ecole Polytechnique de l'Université Côte d'Azur.
***
# Modélisation de la propagation d'une épidémie

## Présentation
Ce projet a été réalisé avec le langage Python.

### Objectifs :
* Modélisation de la propagation d’un agent infectieux au sein d’une population est un phénomène dynamique.
* Effectifs d’individus sains et malades évoluent dans le temps, en fonction des contacts au cours desquels cet agent passe d’un individu infecté à un individu sain non immunisé, l’infectant à son tour.
* Modélisation par des équations différentielles.
* But : Résoudre numériquement ces équations en mettant en oeuvre les méthodes numériques étudiées.

Pour ce projet, un modèle SIR a été utilisé. Ce modèle a la particularité de tenir compte de la présence d’un traitement. Nous avons la population N qui est divisée en quatre compartiments : S, I, T et R tels que : N=S+I+T+R.

S représente les sujets susceptibles d’être infectés, I représente les personnes infectieuses, R représente les personnes dites « rétablies », T représente les traités.

Plusieurs autres paramètres vont entrer en jeu : β représente le nombre de personnes rencontrées par un individu, γ représente le taux de guérison, α représente la quantité d’individu sélectionnés pour être traités, η représente le taux déterminant le passage du compartiment T au R.

![alt text](https://github.com/JulienChoukroun/Modelisation-de-la-propagation-d-une-epidemie-Python/blob/master/Images/SystemeEquations.png "Système d'équations de départ")

![alt text](https://github.com/JulienChoukroun/Modelisation-de-la-propagation-d-une-epidemie-Python/blob/master/Images/Modele.png "Modèle")

Différentes méthodes de résolution de système ont été utilisées, comme la méthode de Newton et la méthode du point fixe avec une méthode d’Euler implicite pour une résolution d’un système d’équations non-linéaires. Ou bien la méthode de Jacobi, la méthode SOR et la méthode de Gauss-Seidel avec une méthode d’Euler explicite pour une résolution d’un système d’équations linéaires.

Des exemples de résultats :

![alt text](https://github.com/JulienChoukroun/Modelisation-de-la-propagation-d-une-epidemie-Python/blob/master/Images/Resultat1.png "Résultats 1")

![alt text](https://github.com/JulienChoukroun/Modelisation-de-la-propagation-d-une-epidemie-Python/blob/master/Images/Resultat2.png "Résultats 2")
