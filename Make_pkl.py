import pandas as pd
import pickle
import numpy as np

columns = ['prcp(mm/day)', 'srad(W/m^2)', 'swe(kg/m^2)', 'tmax(deg c)', 'tmin(deg c)', 'vp(Pa)', 'Radians', 'Year']

data = np.random.random((1000, len(columns)))

df = pd.DataFrame(data, columns=columns)

print(df.head())

df.to_pickle('LSTM_Data.pkl')