# Purpose: train an unsupervised model/s on OLIST data for RFM. 
# Graph the models using basic sns and plt

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Groups: one time big spenders, loyalty, at-risk(of leaving)/ lapsed(no longer purchasing goods)
# possible learning KMeans, Heirarchical, 
# for KMeans use Elbow method (.inertia_ and for loop)