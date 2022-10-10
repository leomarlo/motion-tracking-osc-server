from IGTSpython.code.IGTS import DP_IGTS
import pandas as pd
import os 

fps = 15
fileName = "MariaMovementSequence_xyz_28Sept"
df = pd.read_csv(
    os.path.join("..","data", "csv", fileName + ".csv"), 
    header=0,
    index_col=0)
# df["time"] = 1000 * df.index / fps

DP_TT,_ = DP_IGTS(df.data, 40, 2, 1)

# DP_TT,_ =DP_IGTS(Integ_TS, k,1,1)
print('Dynamic Programming extracted TT >>>' , DP_TT)