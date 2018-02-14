#!/bin/bash

# recursively traverse a directory and convert the input files from certain format to ARFF format
# usage: ROOT=/path/to/utilities ./convert_to_arff.sh format input_folder output_folder(optional)
# arguments: format input_folder output_folder

ROOT='.'

function traverse() {
    # if output folder is given, create output directory
    if [ $4 != -1 ]
    then
        mkdir -p $4
    fi
    # loop through all the files under the input directory
    for file in "$3"/*
    do
        # get the filename of the file, i.e., without directory information
        filename=$(basename $file)
        # if file is not a directory
        if [ ! -d $file ]
        then
            if [[ $filename =~ $input_regex ]]
            then
                if [ $4 != -1 ]
                then
                    cmd="python $ROOT/parsing/classification_format_parser.py --input_format $1 --output_format $2 --input $file --output $4"
                    eval $cmd
                else
                    cmd="python $ROOT/parsing/classification_format_parser.py --input_format $1 --output_format $2 --input $file"
                    eval $cmd
                fi
            else
                echo "skip $file"
                continue
            fi
        # if file is a directory, recurse
        else
            if [ $4 != -1 ]
            then
                traverse $1 $2 $file $4/$filename
            else
                traverse $1 $2 $file -1
            fi
        fi
    done
}

num_args=$#

case $num_args in
2)
    input_format=$1
    output_format=$2
    input_folder=$3
    output_folder=-1
    ;;
3)
    input_format=$1
    output_format=$2
    input_folder=$3
    output_folder=$4
    ;;
*)
    echo $'Error: number of arguments is less than 2 or greater than 3\n'
    ;;
esac

arff_regex='.*\.arff'
ucr_regex='.*\.ucr'
if [ $input_format == "ucr" ]
then
    input_regex=$ucr_regex
fi

if [ $output_format == "arff" ]
then
    output_regex=$arff_regex
fi

traverse $input_format $output_format $input_folder $output_folder
