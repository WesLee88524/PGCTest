#!/bin/bash

mes=$1

branch=$(git branch --show-current)

git add .
git commit -m "${mes}"
git push -u origin ${branch}

echo "add and push ${mes} success!!"