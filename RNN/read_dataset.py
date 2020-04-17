import pandas as pd
import json
import seaborn as sns

PATH = 'News_Category_Dataset_v2.json'

data = []
with open(PATH, 'r') as f:
    headlines = list(map(lambda x: x.strip(), f.readlines()))
    for line in headlines:
        data.append(json.loads(line))

print(data[:50])

df = pd.DataFrame(data)
df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
print(df.head())
print(max([len(h.split(' ')) for h in df.headline]))

import matplotlib.pyplot as plt
from collections import Counter

f, ax = plt.subplots(1,1, figsize=(12,6))

categories = Counter(df.category.values)
print(sorted(categories.values())[:10])
classes = categories.keys()
sns.countplot(x='category', data=df)

# plt.xticks()
ax.xaxis.set_ticklabels(classes, rotation=45, ha='right')
f.tight_layout()
plt.show()



