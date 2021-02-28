import pandas as pd
import os
import numpy as np
import scipy
import math as m
import pickle
import AIS_tools

st_dist = AIS_tools.create_spacetime_distribution()

# save space_time_distributino.pkl
pickle.dump(st_dist, open( "/Volumes/Ocean_Acoustics/AIS_Data/space_time_distribution.pkl", "wb" ) )