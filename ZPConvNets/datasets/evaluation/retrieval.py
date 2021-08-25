import numpy as np
from sklearn.neighbors import KDTree

# calculate mean average precision for modelnet retrieval given a set of features and labels
def modelnet_retrieval_mAP(feats, labels, n=1):
	db = KDTree(feats)
	_, ids = db.query(feats, k=(n+1))
	query_ids = ids[:,1:]

	# Nxn 
	match_labels = labels[:,None].repeat(n,axis=1)
	query_labels = labels[query_ids]
	precision = np.sum(match_labels == query_labels,axis=1).astype(float) / n
	return 100 * precision.mean()