#!/bin/bash

WC1=$(wc -l $1)
cat "$1" | cut -d";" -f $2 | awk '{print tolower($0)}' | sort | uniq >/tmp/uniq.csv
WC2=$(wc -l /tmp/uniq.csv)
echo $WC1
echo $WC2
