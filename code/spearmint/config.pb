language: PYTHON
name:     "title_tag_optimize"


# WEIGHTS[0] = basic weight
# WEIGHTS[1] = tag2tag weight
# WEIGHTS[2] = title2tag weight
variable {
	name: "WEIGHTS"
	type: FLOAT
	size: 3
	min: 0.001
	max: 0.999
}


# THRESHOLD = meta_recommender threshold
variable {
	name: "THRESHOLD"
	type: FLOAT
	size: 1
	min: 0.001
	max: 0.999
}