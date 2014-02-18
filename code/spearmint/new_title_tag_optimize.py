import pyximport; pyximport.install()
from lib.recommender import new_cython_recommender

def title_tag_optimize(weights, threshold):
	'''
	WEIGHTS[0] = basic threshold
	WEIGHTS[1] = tag2tag threshold
	WEIGHTS[2] = title2tag threshold
	THRESHOLD = meta_recommender THRESHOLD
	'''
	f1 = new_cython_recommender.mp_boiler(weights, threshold,
		     num_eng=2)
	return f1

def main(job_id, params):
	print "weights + threshold:", ' '.join([str(x) for x in params])
	assert type(params['threshold']) == float
	return title_tag_optimize(params['weights'], params['threshold'])