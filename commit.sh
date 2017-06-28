#!/bin/bash

# commits a change
git add --all;

echo "please enter the commit message followed by [enter]: "

read msg

git commit -m "$msg"

echo "Hello "$USER". Please enter the name of the upstream remote followed by [ENTER]. Type Enter to push to 'master' "

read remote

if [$remote == "" ]; then
   remote=master
fi

git push -u origin $remote


