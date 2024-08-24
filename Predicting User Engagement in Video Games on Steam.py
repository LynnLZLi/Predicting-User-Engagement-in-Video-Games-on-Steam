# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
## time played prediction ##

# +
import gzip
import json
import pandas as pd
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
from sklearn import linear_model


def readGzippedData(path):
    with gzip.open(path, 'rt') as file:
        return [eval(line) for line in file]

def splitData(data, train_ratio=0.9435):
    n_train = int(len(data) * train_ratio)
    return data[:n_train], data[n_train:]


data_path = 'train.json.gz'
pairs_path = 'pairs_Hours.csv'

train_data = readGzippedData(data_path)
hours_train, hours_valid = splitData(train_data)


user_hours = defaultdict(list)
game_hours = defaultdict(list)
for record in hours_train:
    user, game, hours = record['userID'], record['gameID'], record['hours_transformed']
    user_hours[user].append((game, hours))
    game_hours[game].append((user, hours))

global_avg = sum([record['hours_transformed'] for record in hours_train]) / len(hours_train)

user_bias = defaultdict(float)
game_bias = defaultdict(float)
alpha = global_avg


def updateParameters(lambda_reg):
    new_alpha = sum([r - (user_bias[u] + game_bias[g]) for u, g, r in [(record['userID'], record['gameID'], record['hours_transformed']) for record in hours_train]]) / len(hours_train)
    for u in user_hours:
        user_bias[u] = sum([r - (new_alpha + game_bias[g]) for g, r in user_hours[u]]) / (lambda_reg + len(user_hours[u]))
    for g in game_hours:
        game_bias[g] = sum([r - (new_alpha + user_bias[u]) for u, r in game_hours[g]]) / (lambda_reg + len(game_hours[g]))
    return new_alpha, user_bias, game_bias


lambda_reg = 4.291
for _ in range(10):
    alpha, user_bias, game_bias = updateParameters(lambda_reg)


def calculatePrediction(user, game):
    return alpha + user_bias.get(user, 0) + game_bias.get(game, 0)


pairs_df = pd.read_csv(pairs_path)
predictions = [(u, g, calculatePrediction(u, g)) for u, g in zip(pairs_df['userID'], pairs_df['gameID'])]


predictions_df = pd.DataFrame(predictions, columns=['userID', 'gameID', 'predictedHours'])
predictions_file = 'predictions_Hours.csv'
predictions_df.to_csv(predictions_file, index=False)


print(f"Predictions saved to {predictions_file}")


# +
## would play prediction ##
# -

pip install implicit

import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from implicit import bpr


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)
print(allHours[0])

hoursTrain = allHours[:150000]
hoursValid = allHours[150000:]
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
for u,g,d in hoursTrain:
    r = d['hours_transformed']
    hoursPerUser[u].append((g,r))
    hoursPerItem[g].append((u,r))

# +
# Generate a negative set

userSet = set()
gameSet = set()
playedSet = set()

for u,g,d in allHours:
    userSet.add(u)
    gameSet.add(g)
    playedSet.add((u,g))

lUserSet = list(userSet)
lGameSet = list(gameSet)

notPlayed = set()
for u,g,d in hoursValid:
    #u = random.choice(lUserSet)
    g = random.choice(lGameSet)
    while (u,g) in playedSet or (u,g) in notPlayed:
        g = random.choice(lGameSet)
    notPlayed.add((u,g))

playedValid = set()
for u,g,r in hoursValid:
    playedValid.add((u,g))

# +
userIDs,itemIDs = {},{}

for u,g,d in allHours:
    u,i = d['userID'],d['gameID']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)

nUsers,nItems = len(userIDs),len(itemIDs)
# -

nUsers,nItems

# +
Xui = scipy.sparse.lil_matrix((nUsers, nItems))
for u,g,d in allHours:
    Xui[userIDs[d['userID']],itemIDs[d['gameID']]] = 1
    
Xui_csr = scipy.sparse.csr_matrix(Xui)
# -

model = bpr.BayesianPersonalizedRanking(factors = 5)

model.fit(Xui_csr)

recommended = model.recommend(0, Xui_csr[0])
related = model.similar_items(0)

itemFactors = model.item_factors
userFactors = model.user_factors

# +
gameCount = defaultdict(int)
totalPlayed = 0



for u,g,_ in allHours:
    gameCount[g] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > 1.2* totalPlayed/2: break

# +
import numpy as np

# Load BPR model factors and popularity data structures
# Assume `user_factors`, `item_factors`, `userIDs`, `gameIDs`, `return1` are already defined

# Open the predictions file
predictions = open("predictions_Played.csv", 'w')

with open("pairs_Played.csv") as file:
    for l in file:
        if l.startswith("userID"):
            # Write the header to the predictions file
            predictions.write(l)
            continue
        
        u, g = l.strip().split(',')
        
        # Get the index in the factors matrix for the user and game
        user_idx = userIDs.get(u)
        game_idx = itemIDs.get(g)
        
        # Calculate the BPR score if we have the factors for both user and game
        bpr_score = 0
        if user_idx is not None and game_idx is not None:
            user_vector = userFactors[user_idx]
            game_vector = itemFactors[game_idx]
            bpr_score = np.dot(user_vector, game_vector)
        
        # Determine if the game is popular
        is_popular = g in return1
        
        # Adjusted prediction logic
        bpr_threshold = 0.9  # Adjusted threshold
       
        pred = 1 if is_popular or bpr_score > bpr_threshold else 0  

        # Write the prediction to the file
        predictions.write(u + ',' + g + ',' + str(pred) + '\n')

# Close the predictions file
predictions.close()

