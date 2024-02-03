import pandas as pd
import numpy as np
def topsis(data,weights,impacts,output_file):
  df = data
  df = df.drop(data.columns[0], axis=1)
  mat = np.array(df)
  w = np.array(weights)
  i = np.array(impacts)
  normalized_mat = mat/np.linalg.norm(mat, axis=0)
  weighted_normalized_mat = normalized_mat*w
  if '-' in impacts:
    ideal_best = np.min(weighted_normalized_mat,axis=0)
    ideal_worst = np.max(weighted_normalized_mat,axis=0)
  else:
    ideal_best = np.max(weighted_normalized_mat,axis=0)
    ideal_worst = np.min(weighted_normalized_mat,axis=0)
  eucledian_dist_best = np.linalg.norm(weighted_normalized_mat-ideal_best,axis=1)
  eucledian_dist_worst = np.linalg.norm(weighted_normalized_mat-ideal_worst,axis=1)
  performance_score = eucledian_dist_worst/(eucledian_dist_best + eucledian_dist_worst)
  
  data['Topsis_Score'] = pd.DataFrame(performance_score, columns = ['Topsis_Score'])
  data['Rank'] = data['Topsis_Score'].rank(ascending=False, method='min').astype(int)
  
  return data.to_csv(output_file, index=False)
