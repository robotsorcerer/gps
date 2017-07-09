#!/bin/bash

echo -e "enter iteration to copy from \n"
read from

echo -e "enter itr to copy to \n"
read to

for f in *$from".pkl"; do cp $f "${f/$from/$to}"; done

echo -e "DONE!!!\n"

echo -e "would you like to delete the *$from".pkl" files? (y|n) \n\n"
read resp

if [[ $resp="" ]]; then
	exit $1
fi



