#!/bin/bash
if [ ! -d ./data ]; then
  echo "Create data directory..."
  mkdir -p ./data
fi

for file in "$@"; do
  if which pigz > /dev/null; then
    echo "Unzip $file with pigz..."
    tar -I pigz -xvf $file -C data/
  else
    echo "Unzip $file..."
    tar -xvf $file -C data/
  fi
done
