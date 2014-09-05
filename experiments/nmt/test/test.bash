#!/usr/bin/env bash

# Run this script to test that your current code works as it worked before
# Needs environmental variable GHOG to be set to the location of the repository

set -ux

STATUS="ok"
export PYTHONPATH=$GHOG:$PYTHONPATH

echo "Stage 0: Downloading the data"

DIR='tstwrkspc'
# Until we upload the models to a server the test data is 
# stored in LISA data storage, which means that running
# tests is currently possible only from the LISA lab.
DATAURL='file://localhost/data/lisatmp3/bahdanau/testdata'

mkdir $DIR
cd $DIR

curl $DATAURL/data.tgz >data.tgz
tar xzf data.tgz

echo "Stage 1: Test scoring"

SCORE="$GHOG/experiments/nmt/score.py --mode=batch --src=english.txt --trg=french.txt --allow-unk"
$SCORE --state encdec_state.pkl encdec_model.npz >encdec_scores.txt 2>>log.txt
$SCORE --state search_state.pkl search_model.npz >search_scores.txt 2>>log.txt
if ! diff encdec_scores.txt $GHOG/experiments/nmt/test/encdec_scores.txt
then
    echo "TEST FAILED: RNNencdec scores changed!"
    STATUS="failed"
fi    
if ! diff search_scores.txt $GHOG/experiments/nmt/test/search_scores.txt
then
    echo "TEST FAILED: RNNsearch scores changed!"
    STATUS="failed"
fi    

if [ $STATUS = "ok" ]
then
    echo "Stage -1: Cleaning up"
    cd ../
    rm -rf $DIR
else
    echo "There are failed tests, check $DIR for the new outputs"    
fi  
