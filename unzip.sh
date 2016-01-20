#!/bin/bash
if [ ! -d ./data ]; then
  echo "Create data directory..."
  mkdir -p ./data
fi

echo "Unzip cnn.tgz..."
if [ type "pigz" &> /dev/null ]; then
  tar -xvf -C data/ | pigz > cnn.tgz
else
  tar -xzvf cnn.tgz -C data/
fi

echo "Unzip cnn.tgz..."
if [ type "pigz" &> /dev/null ]; then
  tar -xvf -C data/ | pigz > dailymail.tgz
else
  tar -xzvf dailymail.tgz -C data/
fi
