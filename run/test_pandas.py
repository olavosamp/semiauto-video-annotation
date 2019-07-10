import pandas as pd
import numpy as np

## Exp1
# input:  dict of lists
# method: from_dict
print("\nExp1")
entry1L = {'col1': ['var1'],
           'col2': [111],
           'col3': ['wololo']}

df = pd.DataFrame.from_dict(entry1L)

# append
entry2L = {'col1': ['var2'],
           'col2': [222],
           'col3': ['wololo']}
df2 = pd.DataFrame.from_dict(entry2L)
df = df.append(df2, sort=False,
                        ignore_index=False).reset_index(drop=True)
print(df)

## Exp2
# input:  list of dict of lists
# method: from_dict
print("\nExp2")
entry1 = {'col1':'var1',
         'col2': 111,
         'col3': 'wololo'}


# append
entry2 = {'col1':'var2',
         'col2': 222,
         'col3': 'wololo'}

entryList = [entry1, entry2]
# print(np.shape(entryList))
# print(np.shape(np.squeeze(entryList)))
# exit()
df = pd.DataFrame(entryList)
# df2 = pd.DataFrame(entry2)
# df = df.append(df2, sort=False,
#                         ignore_index=False).reset_index(drop=True)
print(df)
