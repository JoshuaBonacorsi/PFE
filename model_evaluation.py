import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import silhouette_score, average_precision_score, f1_score, precision_score,recall_score

def evaluate(df,anomalies_predicted):
            
            silhouette = round(silhouette_score(df[['value']], df[anomalies_predicted]),2)
            auprc = round(average_precision_score(df['class'], df[anomalies_predicted]),2)
            f1 = round(f1_score(df['class'], df[anomalies_predicted]),2)
            precision = round(precision_score(df['class'], df[anomalies_predicted]),2)
            recall = round(recall_score(df['class'], df[anomalies_predicted]),2)

            return (silhouette,auprc,f1,precision,recall)