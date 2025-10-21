#!/usr/bin/env bash

echo "Installing Hugo tool"

latest_version=$(curl -s "https://api.github.com/repos/gohugoio/hugo/releases" \
  | jq -r '.[].tag_name' \
  | sort -V \
  | tail -n 1)

echo "Latest Hugo version: $latest_version"
url="https://github.com/gohugoio/hugo/releases/download/${latest_version}/hugo_extended_${latest_version#v}_linux-amd64.tar.gz"

echo "Downloading Hugo from $url"
wget -q "$url" -O hugo.tar.gz

echo "Extracting Hugo binary"
tar -xzf hugo.tar.gz hugo

echo "Cleaning up"
rm hugo.tar.gz

echo "Run the tool with ./hugo"
