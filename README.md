# Examen BentoML

## 1. Présentation du projet :

Le projet consiste en la création d'un Bento conteneurisé avec Docker qui contient un environnement 
complet avec mise à disposition (serving) d'un modèle de prédiction *d'admission d'étudiants aux universités*.
Le modèle est accessible via API, suite à une authentification via user/password valide.

Les livrables fournis sont :
* Fichier README ==> *self*
* Image du container ==> *bento_image.tar*
* Fichier des tests unitaires ==> *api_test.py*

Voici la structure du Bento 🍱 :

```bash       
├── examen_bentoml          
│   ├── data       
│   │   ├── processed      
│   │   └── raw           
│   ├── lib
│   │   └── custom_libraries.py     
│   └── src  
│       ├── api
│       │   └── service.py      
│       ├── data
│       │   └── prepare_data.py
│       └── models    
│           └── train_model.py       

```

---
## 2. Endpoints disponibles :

* Home : 
    - /home (get) : Infos générales
    `curl -X 'GET' 'http://localhost:3000/home'`

* Test :
    - /verify (get) : Vérification fonctionnement de l'API
    `curl -X 'GET' 'http://localhost:3000/verify'`

    - /auth-test (post) : Endpoint dédié aux tests unitaires
    `no parameters`

* Login :
    - /login (post) : Authentification par user + password ==> **renvoie un token**
    ```bash
        curl -X 'POST' \
        'http://localhost:3000/login' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "username": "user123",
        "password": "password123"
        }'
    ```

* Predict :
    - /predict (post) : Renvoi une inférence en fonction des features envoyées
    **MODIFIER LE TOKEN PAR CELUI RECU SUR L'ENDPOINT /LOGIN**

    ```bash
        curl -X 'POST' \
        'http://localhost:3000/predict' \
        -H 'accept: application/json' \
        -H 'Authorization: Bearer **TOKEN**' \
        -H 'Content-Type: application/json' \
        -d '{
        "GRE_Score": 331,
        "TOEFL_Score": 120,
        "University_Rating": 3,
        "SOP": 4,
        "LOR_": 4,
        "CGPA": 8.96,
        "Research": 1
        }'
    ```


---
## 3. Mise en route du *container* :

Une fois l'image récupérée : **bento_image.tar**
on la charge sur la machine avec la commande :

`docker load -i bento_image.tar`

On lance le container :

`docker run --rm -p 3000:3000 admission_api:latest`

(les warnings apparaissant sont normaux, ils nous indiquent que des modifications 
dans les versions > 1.4 de Bentoml ont été apportées, notamment `bentoml.Service`)

---
## 4. Commandes des tests unitaires :

Afin de tester le code des services fournis par Bentoml, nous pouvons lancer le script Pytest à partir du fichier *api_test.py* :

`pytest api_test.py -W ignore`

(l'ajout de `-W ignore` limite la liste des warnings)

Vous devez trouver 9 tests passed comme ceci :

```bash
============================================ 9 passed, 3 warnings in 3.09s =============================================
```
