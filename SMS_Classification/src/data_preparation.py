import os
import pandas as pd
import glob
from util import Util
import numpy as np
import random as random
util = Util()
os.chdir("E://Study//GitLab//SMS_Classification//data")
cwd = os.getcwd()
files = glob.glob(cwd+"//*.csv")

for file in files:
    consolidated_data = pd.read_csv(file,encoding="latin-1")

util.show_column_names(consolidated_data)

# dataframe after dropping the unwanted drop_columns
consolidated_data = util.drop_columns(consolidated_data)

# dataframe after renaming the columnNames
consolidated_data = util.rename_columns(consolidated_data,["category","message"])
#print(consolidated_data.head())
#np.random.rand([d0], [d1], [...], [dn])

# now let us split the data into train and test data to get prepare the data for training and testing
training_index = random.sample(consolidated_data.index.tolist(),int(0.80*consolidated_data.shape[0]))
testing_index = [index not in training_index for index in consolidated_data.index]

training_data = consolidated_data.iloc[training_index,]
testing_data = consolidated_data.iloc[testing_index,]

training_dir = cwd+"//training_data.csv"
testing_dir = cwd+"//testing_data.csv"

training_data.to_csv(cwd+"//training_data.csv",index=False)
testing_data.to_csv(cwd+"//testing_data.csv",index=False)
