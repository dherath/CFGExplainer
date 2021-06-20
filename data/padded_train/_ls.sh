#!/bin/bash

#printf "Git-source: "
#read -r gitsource

# declare -a count=1
# declare -a names=()

# echo "A53: clone started : missing anthonyjinete45"
for folder in *;
do
    if [ -d "$folder" ];
       then
	   cd $folder
	   ls > 0_list.txt
	   cd ..
    fi
done

echo "complete"
