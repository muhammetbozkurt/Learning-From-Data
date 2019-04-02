import classifiers

def main():
	print("\n\nHW1 part 1:\n")
	reg = classifiers.Regression()
	reg.creating_folds()
	reg.evaluate_model()
	print("\n\nHW1 part 2:\n")
	disc = classifiers.Discriminant()
	disc.cal_mean_cov()
	disc.covariance()
	disc.meanp()
	disc.visualize()
	acc = disc.test_model()
	print("\n\naccuracy: ",acc,"\n\n")

if __name__ == '__main__':
	main()
