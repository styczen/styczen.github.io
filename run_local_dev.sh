#!/bin/sh

NAME=website_local_env
if [ $(docker ps -aq --filter name=$NAME | wc -l) -eq 1 ];
then
    docker start -i $NAME
else
    docker run -p 4000:4000 -v ${PWD}:/srv/jekyll --name $NAME jekyll/jekyll:3.5 jekyll serve --watch --drafts --incremental
fi

