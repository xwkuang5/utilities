#!/bin/bash

# This script extract hypnogram annotations for all the *.edf+ files under the input folder using the rdedfann command
# usage: ./extract_hyp_from_edf+.sh /path/to/folder

regex='(.*).-Hypnogram\.edf$'

num_args=$#

case $num_args in
1)
    input_folder=$1
    ;;
*)
    echo $'Error: number of arguments is not 1\n'
    ;;
esac

for file in $input_folder/*
do
    filename=$(basename $file)
    if [[ $filename =~ $regex ]]
    then
        out_filename=$(dirname $file)/${BASH_REMATCH[1]}0-PSG.txt
        cmd="rdedfann -r $file > $out_filename"
        echo $cmd
        eval $cmd
    fi
done
