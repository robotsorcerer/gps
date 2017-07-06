#!/bin/bash

echo -e "enter the experiment to copy from\n\n"
read src_expt

echo -e "enter the experiment to copy TO\n\n"
read dest_expt

if [ -d !$dest_expt ]; then
  echo -e "please create $dest_expt using the gps main file.\n\n";
  exit 1
fi

cp "$src_expt/hyperparams.py" $dest_expt
wait

on=on_policy
off=off_policy

cp -R "$src_expt$on/." $dest_expt
wait


cp -R "$src_expt$off/." $dest_expt
wait

echo -e "enter the itr you want to copy from\n\n"

read itr

echo -e "enter the itr you want to copy to\n\n"

read dest_itr

cd "$dest_expt/data_files"
for f in *$itr.pkl; do mv $f "${f/$itr/$dest_itr}"; done

#remove unneeded pkls
# rm !(*"$dest_itr")

cd ../

# rm policy*

cd ../

echo "done!!!"
