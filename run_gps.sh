#!/bin/bash
gps=python/gps/gps_main.py

echo -e "please enter expt name: \n"
read name

echo -e "running in close loop with antagonist today?[y|n] \n\n"
read closeloop

if [[ $closeloop=="y" ]]; then
	python $gps $name -c
elif [[ $closeloop=="n" ]]; then
	echo -e "are you testing a pretrained policy?\n\n"
	read testing
	
	if [[ $testing=="y" ]]; then
		python $gps $name -p 2000
	else
		python $gps $name 
	fi
fi
