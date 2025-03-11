#!/bin/bash

# Usage: ./experiments.sh {n} {program} {argc1} {argc2} ...
# where n is how many times repeate and the experiments
# program is the program to run
# argc1, argc2, ... are the arguments to pass to the program

# All the output must be saved in a file called output.csv

n=$1
program=$2
argc=("${@:3}")

# Create or clear the output CSV file add the name of program but remove ./
output_file="${program:2}.csv"
# run, output and first argc at the head of files
echo "Run,Output,${argc[0]}" > $output_file
for i in $(seq 1 $n)
do
    echo "Running $program with arguments ${argc[@]}"
    output=$($program ${argc[@]})
    echo "$i,$output" >> $output_file
done

