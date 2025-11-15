import DataFunctions, TrainingFunctions, TestingFunctions
import pandas as pd

path = "../data/all_stocks_5yr.csv"
features = ["open", "high", "low", "close", "volume"]

df = DataFunctions.getDataframe(path)
df = df[features]

history = TrainingFunctions.trainTorchLSTM(df, name="testmodel", epochs=1, pred_col=3)
