#!/bin/bash

run() {
    echo "▶ Running $1"
    bash "$1"
    if [ $? -ne 0 ]; then
        echo "❌ Failed: $1"
        exit 1
    fi
}

run script1.sh
run script2.sh
run script3.sh

echo "✅ All done"
