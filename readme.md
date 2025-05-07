# How to use
For now you can use the train_and_test.py script with a config file to train and test a model on a dataset with or without a preprocessing, with or without data augmentation with choosen preprocessing, factorial method and generator or EMSA augmentation.

## General
To run use :
```
python train_and_test.py --conf config.json
```

In the config file you can precise the model to use, the learning rate LR, the dataset, the preprocessing or the augmentation you want, for exemple : 
```
{
  "model" : "CuiNet",
  "LR" : 0.01,
  "dataset" :
    {
      "name" : "wheat"
    },
  "augmentation" : 
    {
      "preprocessing" : 
        {
          "name" : "EMSC",
          "order" : 3
        },
      "factorialMethod" : 
        {
          "name" : "PCA",
          "nb_comp" : "all"
        },
      "generator" : 
        {
          "name" : "localGaussian",
          "alpha" : 0.1
        }
    }
}
```
If you don't precise LR, optuna will optimize those hyper-parameters and save the updated config.
## Datasets
For now only wheat, mango and corn are implemented. If you want to use the corn dataset you have to precise wich target you want between moisture, protein, oil and starch_value : 
```
"dataset" :
{
    "name" : "corn",
    "target" : "protein"
}
```
## Preprocessing
If you want to use a preprocessing you can use the following, for now only EMSC and SNV are available :
```
{
  "model" : "CuiNet",
  "LR" : 0.015,
  "dataset" :
    {
      "name" : "wheat"
    },
  "preprocessing" : 
    {
        "name" : "SNV"
    }
}
```
## Augmentation
### Preprocessing
The preprocessing for augmentation are the same than classical preprocessing (EMSC and SNV). 
### Factorial method
For the factorial method, you can use PCA or MVN.
For the PCA, you can precise the number of coponents you want or the proportion of the variance you want to explain : 
```
"factorialMethod" : 
    {
        "name" : "PCA",
        "nb_comp" : "all"
    }
```
```
"factorialMethod" : 
    {
        "name" : "PCA",
        "nb_comp" : 20
    }
```
```
"factorialMethod" : 
    {
        "name" : "PCA",
        "var_explained" : 0.94
    }
```
### Generator
Gaussian, localGaussian and MVN generators are available. If you use localGaussian generator please precise the alpha parameter. If you want to use MVN you have to use it both for factorial method and generator ex : 
```
{
  "model" : "CuiNet",
  "LR" : 0.01,
  "dataset" :
    {
      "name" : "wheat"
    },
  "augmentation" : 
    {
      "preprocessing" : 
        {
          "name" : "EMSC",
          "order" : 3
        },
      "factorialMethod" : 
        {
          "name" : "MVN",
        },
      "generator" : "MVN"
    }
  
}
``` 





## remaining TODO : 
1. réduire le nombre d'epochs
1. trouver les bons params ViT ?
1. reproduire l'EMSA
1. réfléchir à un jdd synthétisé
1. tester de ne pas corriger le spectre
2. mettre en place l'expérience : pouvoir travailler avec des sous-parties des jeux de données, faire des fichiers bashs qui lancent des entraînements en parallèle 
2. ? coder une augmentation aléatoire ?
5. essayer de travailler sur les résidus d'une PLS ? x = x - xpls avec xpls étant x transformé en nb de composantes puis de nouveau dans l'espace d'origine ? Impossible, pk ?
5. travailler sur l'orthogonalisation à y
5. travailler sur l'échantillonnage ?

Bugs : 
1. mettre toutes les composantes ou laisser le comportement par défaut (utiliser toutes les composantes) conduit à enregistrer 2 fichiers différents

Observations : 
1. CuiNet est très sensible à l'init. 