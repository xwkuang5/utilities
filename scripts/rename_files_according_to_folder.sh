#!/bin/bash

# This script is used to rename files under a directory
# usage: ./rename.sh folder mode
# if forward mode is used, ECG/TRAIN.arff => ECG/ECG_TRAIN.arff
# if backward mode is used, ECG/ECG_TRAIN.arff => ECG/TRAIN.arff
# Optionally change regex, prefix and suffix for other renaming
# Note that bash variable adopts dynamic scoping. Think carefully about whether you want "local" variable or not.

regex='(TRAIN|TEST)'
prefix=''
suffix='\.arff'

match_regex=$prefix$regex$suffix

case $# in
2)
    directory=$1
    mode=$2
    ;;
*)
    echo "Errors with input arguments"
    echo "./rename.sh folder mode"
    exit 64
    ;;
esac

function traverse() {
    local directory=$1
    local mode=$2
    local folder_base_name=$(basename $directory)
    for item in $directory/*
    do
        if [ -d $item ]
        then
            cmd="traverse $item $mode"
            echo $cmd
            eval $cmd
        else
            file_base_name=$(basename $item)
            if [ "$mode" == "forward" ]
            then
                if [[ $file_base_name =~ $match_regex ]]
                then
                    cmd="mv $item $directory/$folder_base_name"_"$file_base_name"
                    echo $cmd
                    eval $cmd
                fi
            elif [ "$mode" == "backward" ]
            then
                modified_regex=$folder_base_name'_'$match_regex
                if [[ $file_base_name =~ $modified_regex ]]
                then
                    name=${BASH_REMATCH[1]}
                    # note how $regex is not used
                    cmd="mv $item $directory/$prefix$name$suffix"
                    echo $cmd
                    eval $cmd
                fi
            fi
        fi
    done
}

traverse $directory $mode
