#!/bin/bash

# This file is to avoid your system being fired up with models.
DIR="./models"
KEEP=5
WARNING_S=30
CHECK_INTERVAL=100

cd "$DIR" || exit

while true; do
    mapfile -t files < <(ls -1t model_ep*.mpk 2>/dev/null)

    if [ ${#files[@]} -gt $KEEP ]; then
        for ((i=KEEP; i<${#files[@]}; i++)); do
            file="${files[$i]}"
            echo "Hey, the $file is gonna be deleted!"
            sleep $WARNING_S
            rm -f "$file"
            echo "Deleted: $file"
        done
    else
        sleep $CHECK_INTERVAL
    fi
done

