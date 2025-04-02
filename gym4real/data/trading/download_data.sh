#!/bin/bash

# This is a simple script to download klines by given parameters.
# Source: https://github.com/binance/binance-public-data

symbols=("BTCUSDT") # add symbols here to download
intervals=("1m" "1h")
years=("2017" "2018" "2019" "2020" "2021" "2022")
months=(01 02 03 04 05 06 07 08 09 10 11 12)
minute_folder="./BTCUSDT_minutes_data"
hour_folder="./BTCUSDT_hour_data"

baseurl="https://data.binance.vision/data/spot/monthly/klines"

for symbol in ${symbols[@]}; do
  for interval in ${intervals[@]}; do
    for year in ${years[@]}; do
      for month in ${months[@]}; do
        url="${baseurl}/${symbol}/${interval}/${symbol}-${interval}-${year}-${month}.zip"
        echo $url
        echo $interval

        if [ $interval == "1m" ]; then
          save_folder=minute_folder
        elif [ $interval == "1h" ]; then
          save_folder=hour_folder
        else
            echo "Error saving folder $interval"
        fi

        if [[ ! -e $save_folder ]]; then
            mkdir -p $save_folder
        fi

        echo $save_folder

        response=$(wget --server-response -q ${url} 2>&1  -O "$save_folder/$(basename "$url")" | awk 'NR==1{print $2}')
        if [ ${response} == '404' ]; then
          echo "File not exist: ${url}"
        else
          echo "downloaded: ${url}"
        fi
      done
    done
  done
done