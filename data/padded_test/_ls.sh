#!/bin/bash

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
