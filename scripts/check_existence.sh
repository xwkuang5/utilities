#!/bin/bash

# regex to search
regex='.*\.apk$'

existRegex() {
  if [ ! -d "$1" ]
  then
    if [[ "$1" =~ $regex ]]
    then
      # if is file and matches
      echo 1
      exit
    else
      # if is file and does not match
      echo 0
      exit
    fi
  else
    # if is directory check files inside directory
    for file in "$1"/*
    do
      ret=$(existRegex $file)
      if [[ $ret == 1 ]]
      then
        # if at least one file matches, early exist
        echo 1
        exit
      else
        continue
      fi
    done
    # no match, exit
    echo 0
    exit
  fi
}

for folder in *
do
  ret=$(existRegex $folder)
  # if not exist
  if [[ $ret == 0 ]]
  then
    echo $folder
  fi
done
