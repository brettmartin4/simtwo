import numpy as np
import pandas as pd

df = pd.read_csv("D:\simulator2\data\consolidated_weather_timesync.csv", usecols=["t_sec", "temperature_x", "clock_error"])
#df.info()

print(df.describe())

