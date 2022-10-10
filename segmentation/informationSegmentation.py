import pandas as pd
import os

fileName = "MariaMovementSequence_xyz_27Sept"
fps = 15

df = pd.read_csv(
    os.path.join(os.getcwd(),"data", "csv", fileName + ".csv"), 
    header=0,
    index_col=0)
df["time"] = 1000 * df.index / fps
print(df.head(10))