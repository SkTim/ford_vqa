for i in 1 2 3 4 5 6 7 8 9 10
do
	python preprocess.py train $i
	python preprocess.py test $i
done
