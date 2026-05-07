import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier

# 📊 dane (kilka sezonów)
urls = [
    "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
]

df = pd.concat([pd.read_csv(u) for u in urls])

df = df[["HomeTeam","AwayTeam","FTHG","FTAG"]].dropna()
df.columns = ["home","away","hg","ag"]

# 🎯 wynik
def result(r):
    if r.hg > r.ag:
        return 2
    elif r.hg < r.ag:
        return 0
    return 1

df["y"] = df.apply(result, axis=1)

# 🧠 ELO
elo = defaultdict(lambda: 1500)

def expected(a,b):
    return 1/(1+10**((b-a)/400))

for _,r in df.iterrows():
    h,a=r.home,r.away

    eh=expected(elo[h],elo[a])

    if r.hg>r.ag:
        elo[h]+=20*(1-eh)
        elo[a]+=20*(0-eh)
    elif r.hg<r.ag:
        elo[h]+=20*(0-eh)
        elo[a]+=20*(1-eh)
    else:
        elo[h]+=20*(0.5-eh)
        elo[a]+=20*(0.5-eh)

# 📊 features
X=[]
y=[]

for _,r in df.iterrows():
    X.append([elo[r.home]-elo[r.away]])
    y.append(r.y)

X=pd.DataFrame(X)

# 🤖 ML
model=GradientBoostingClassifier()
model.fit(X,y)

teams=list(set(df.home))

def predict(home,away):

    p=model.predict_proba([[elo[home]-elo[away]]])[0]

    return {
        "home_win":round(p[2]*100,2),
        "draw":round(p[1]*100,2),
        "away_win":round(p[0]*100,2)
    }
