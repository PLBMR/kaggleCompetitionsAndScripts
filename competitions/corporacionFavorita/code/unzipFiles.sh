#!/bin/bash
#unzipFiles.sh
#unzips our particular files for us

#install 7z unarchiver

brew install p7zip

#then run through archive files

rawDirFiles=../data/raw/* ;

for f in $rawDirFiles
do  
    if [[ $f == *'sample_submission'* ]]; then
        #is already processed
        7z x $f -o../data/processed
    else
        #not entirely processed
        7z x $f -o../data/preprocessed
    fi
done
