#!/bin/bash

#Clearing output file before writing to it
> test_data.txt

#Running sample executions of program with default parameters
#The number of sample executions is given as input when script is called
for ((i=1;i<=$1;i++))
do
    make run 
done

./test $1 $2

echo "--------------------------------------------------------------------------------------------------"

#Clearing output file before writing to it
> test_data.txt

#Running sample executions of program with an Array Dimension of 10000
for ((i=1;i<=$1;i++))
do
    make run D=10000
done

./test $1 $2

echo "--------------------------------------------------------------------------------------------------"

#Clearing output file before writing to it
> test_data.txt

#Running sample executions of program with a Zero Percentage of 0
for ((i=1;i<=$1;i++))
do
    make run Z=0
done

./test $1 $2

echo "--------------------------------------------------------------------------------------------------"

#Clearing output file before writing to it
> test_data.txt

#Running sample executions of program with a Zero Percentage of 99
for ((i=1;i<=$1;i++))
do
    make run Z=99
done

./test $1 $2

echo "--------------------------------------------------------------------------------------------------"

#Clearing output file before writing to it
> test_data.txt

#Running sample executions of program with 1 Loop for multiplication
for ((i=1;i<=$1;i++))
do
    make run R=1
done

./test $1 $2

echo "--------------------------------------------------------------------------------------------------"

#Clearing output file before writing to it
> test_data.txt

#Running sample executions of program with 20 Loops for multiplication
for ((i=1;i<=$1;i++))
do
    make run R=20
done

./test $1 $2

echo "--------------------------------------------------------------------------------------------------"

#Clearing output file before writing to it
> test_data.txt

#Running sample executions of program with 8 nodes
for ((i=1;i<=$1;i++))
do
    make run N=8
done

./test $1 $2

echo "--------------------------------------------------------------------------------------------------"

#Clearing output file before writing to it
> test_data.txt

#Running sample executions of program with 16 nodes
for ((i=1;i<=$1;i++))
do
    make run N=16
done

./test $1 $2
