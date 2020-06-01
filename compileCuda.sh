#!/bin/bash

name=$1

nvcc -o $name $name.cu -lsfml-graphics -lsfml-window -lsfml-system
