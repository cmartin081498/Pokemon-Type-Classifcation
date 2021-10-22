# Pokemon-Type-Classification
## Project Idea
Decied what classification method works best for Pokemon classifying pokemon's type based on the pokemons leanrable movesets. The pokemon_data does take some time to run

## Requirments/Dependencies
python 3.6
scikit-learn 0.23.2
numpy 1.19.1
pokebase 1.3.0

## Usage 
Run pokemon_data.py this will create a JSON pokemon dataset. Set the varibles to decied how many pokemon you want and how many moves are you taking into account
Run project_KNN.py this will train and test KNN and return the best estimator, micro recall, micro f1 score, hamming loss score and hamming score

## Note
There is a pokemon Json file that contains all pokemon from Gen 1 to Gen 7 and all the possible moves they could learn.
