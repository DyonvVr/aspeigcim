for item in eigv_F/input/H2_w*
do
	echo $item
	python3 main.py $item "6-31G"
done
