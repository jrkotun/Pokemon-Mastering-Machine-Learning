# Pokémon Mastering Machine Learning

## Motivation
Throughout my data science bootcamp with Nashville Software School, we studied and implemented various machine learning models. Through our machine learning projects I spent a lot of my time focusing on understanding the data and the domain in which it came from (i.e., healthcare, home mortgages, trucking, etc.). With my capstone project, I wanted to take a deeper dive into machine learning modeling using a dataset where I already have a high level of domain knowledge. I decided to do my capstone project on a video game that I have played my entire life, Pokémon.

## Project Purpose
The purpose of this project was to gather data and build models to answer the below questions:
- Can I build a machine learning model that uses each Pokémon's stats to predict if it is a legendary Pokémon or not?
- Can I build a machine learning model that uses each Pokémon's stats to predict what type it is?

## Capstone Presentation
Capstone Presentation Link: https://sites.google.com/view/pokmon-mastering-machine-learn/title-page

The above link goes to the Google site used for the presentation of my Capstone to my professors at Nashville Software School.

## Data Set
The data used in this project was scraped from the below websites utilizing Python’s Requests and Beautiful Soup libraries:
- Pokémon Database (https://pokemondb.net)
- Bulbapedia (https://bulbapedia.bulbagarden.net/wiki/Main_Page)
- Serebii (https://www.serebii.net/index2.shtml)
 
The Pokémon for this data set were from Pokémon generations 1 through 8.

For the Pokémon Legendary prediction problem, the below features needed to be scraped or calculated:
- Summation and Average of All Stats
- Each Individual Stat
- Catch Rate
- Legendary Status (True or False)

For the Pokémon type prediction problem, the below features needed to be scraped or calculated:
- Primary and Secondary Types
- Egg Groups
- Proportions of Move Types in Level Up Moveset

Pokémon Excluded from Data Set:
- Certain Pokémon transform during battle (via Mega Evolution, abilities, etc.), changing their stats.
- Some Pokémon have multiple forms that are identical in stats and moveset, therefore acting as duplicates in the data set.
- Each of these transformations and duplicate forms were represented as separate rows in the data set. They were removed to improve model accuracy.

Post cleaning, there were 979 unique Pokémon within the data set. Data set included the following column/information for each Pokémon:
- Pokédex Number
- Pokémon Name
- Primary and Secondary Types
- Summation and Average of All Stats
- Each Individual Stat
- Generation (When Pokémon Was Introduced)
- Legendary Status (True or False)
- Catch Rate
- Egg Groups
- Proportions of Move Types in Level Up Moveset 

## EDA Insights
### General EDA
- The dataset includes 874 non-Legendary Pokémon and 105 Legendary Pokémon. There should be enough samples between non-Legendary and Legendary Pokémon to properly train the Legendary prediction model.
- Frequency of Pokémon types range from 144 Water types to 50 Ice types. No type has a drastically low number of samples, so training an effective model shouldn't be an issue.
- Frequency of Pokémon egg groups range from 275 in the Field group and 1 in Ditto.

### EDA for Legendary Prediction
- At least 50% of Legendary Pokémon have higher total stats than non-Legendary Pokémon as seen from the boxplot between the two.
- Almost all Legendary Pokémon have a lower catch rate than 50% of non-Legendary Pokémon.

### EDA for Type Prediction
- There are no moderate or strong correlations between a Pokémon's types and their stats. The best correlations were between the Defense stat and the Rock and Steel types, 0.22 and 0.21 respectively.
- There are a few egg groups that lean towards one Pokémon type versus others:
    - Amorphous Egg Group: 34.65% Ghost type
    - Bug Egg Group: 53.21% Bug type
    - Ditto Egg Group: 100% Normal type
    - Dragon Egg Group: 38.89% Dragon type
    - Fairy Egg Group: 32.61% Fairy type
    - Flying Egg Group: 48.51% Flying type
    - Grass Egg Group: 62.81% Grass type
    - Human-like: 30.43% Fighting type
    - Water 1: 53.85% Water type 
    - Water 2: 57.45% water type
    - Water 3: 38.33% water type
- The higher the proportions of a move type in a Pokémon's Level up moveset the more likely the Pokémon is of that same type. This doesn't hold up true for one Pokémon type, the Normal type. Normal type moves can be learned by all Pokémon types and at different frequencies. Due to this Normal type Pokémon may not be accurately predicted with the moveset percentages.

## Machine Learning Model Results
### Methodology for Model Building
1. Feature Engineering (if needed)
2. Train and test below models for each prediction problem:
    - Decision Tree
    - K-Nearest Neighbors
    - Logistic Regression
    - Naive Bayes
    - Neural Network
    - Random Forest
    - Support Vectors
3.  Choose best performing model and tune

### Predicting Legendary Status
#### Features:
- Stats (Total, Average, Individual)
- Catch Rate

#### Best performing model: Random Forest (97.96% accuracy, 90.74% F1-score)

#### Hyperparameters Tuned using RandomSearchCV: 
- Number of Decision Trees in the Forest: Increase number of trees to make sure all features are covered by model and reduce model error.
- Criteria to Split On: Try different functions to test different node splits.
- Max Depth of the Individual Trees: Deeper trees mean more information about the data is considered possibly leading to better predictions.
- Number of Random Features to Consider at Each Split: Test different number of features at each split to see which results in best results.
- Percentage of the Training Data Used to Train Each Tree (bootstrap): Manipulate how much of the data is used to train each tree.

#### Hyperparameter tuning results in a 0% increase in accuracy and F1-score

### Predicting Pokémon Types
#### Features:
- Egg Groups
- Proportions of Move Types in Level Up Moveset
- Egg group and Pokémon types encoded using MultiLabelBinarizer

#### Best performing model: Neural Network (67.01% accuracy, 81.86% F1-score)

#### Hyperparameters Tuned using RandomSearchCV: 
- Number of Hidden Layers and Number of Neurons per Layer = Keep hidden layer to one but vary number of neurons to see if we can better fit data and improve accuracy.
- Activation Function = Since this is a non-linear classification, try out different activation functions to manipulate the weights as they are leaving neurons. 
- Solver = Choose algorithm for weight optimization across nodes. 
- Learning Rate = Stabilize training process by picking a learning rate that helps the network converge to an output. 
- Momentum = Control speed of gradient descent. Improve training time while maintaining accuracy. 
- Number of Epochs = Purpose is to increase the number of times the whole training set is shown to the network while training. 
- Batch Size = Control number of minibatches that will be used to train the network.

#### Hyperparameter tuning results: 69.73% accuracy (2.72% increase) and 83.39% (1.53% increase) F1-score

## Further/Next Steps for Project
- Utilize GridSearchCV to find best parameters for each model
- Determine method to stratify train test split for type prediction
- Increase number of hidden layers for Neural Network for type prediction
- Explore additional features to use in models:
    - Pokémon abilities
    - Pokémon color schemes
- Use these models on new Pokémon in Pokémon Scarlet and Violet once released in late 2022
