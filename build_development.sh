#!/bin/bash

docker run --name jekyll_server --rm -v "$PWD:/srv/jekyll" -p 4000:4000 jekyll/jekyll jekyll serve

