#!/bin/bash
wget -P data/ https://archive.org/compress/moby_dick_librivox/formats=LIBRIVOX%20APPLE%20AUDIOBOOK,JPEG,TEXT%20PDF,JPEG%20THUMB,ITEM%20TILE,UNKNOWN,STORJ%20UPLOAD%20TRIGGER,ARCHIVE%20BITTORRENT,METADATA,128KBPS%20MP3
unzip -d data/ data/*.zip
find data/ ! -name '*.mp3' -type f -delete
