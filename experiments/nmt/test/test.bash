#!/usr/bin/env bash

# Run this script to test that your current code works as it worked before
# Needs environmental variable GHOG to be set to the location of the repository.
#
# At first run will create a directory test_workspace to which test data will
# be downloaded.

[ "$DEBUG" = 1 ] && set -x
set -u

STATUS="ok"
export PYTHONPATH=$GHOG:$PYTHONPATH
DIFF=diff
NUMDIFF=$GHOG/experiments/nmt/test/diff.py

function check_result() {
    if ! $1 $2 $GHOG/experiments/nmt/test/$2
    then
        echo $3
        STATUS="failed"
    else
        echo "OK"
    fi        
}

echo "Stage 0: Preparing the workspace"

DIR='test_workspace'
# Until we upload the models to a server the test data is 
# stored in LISA data storage, which means that running
# tests is currently possible only from the LISA lab.
DATAURL='file://localhost/data/lisatmp3/bahdanau/testdata'

if [ ! -d $DIR ]
then    
    echo "No test workspace found; creating a new one"
    mkdir $DIR
    cd $DIR
    curl $DATAURL/data.tgz >data.tgz
    tar xzf data.tgz
else    
    cd $DIR
    rm log.txt
fi    

echo "Stage 1: Test scoring"

SCORE="$GHOG/experiments/nmt/score.py --mode=batch --src=english.txt --trg=french.txt --allow-unk"
echo "Score with RNNencdec"
$SCORE --state encdec_state.pkl encdec_model.npz >encdec_scores.txt 2>>log.txt
check_result $NUMDIFF encdec_scores.txt "RNNencdec scores changed!"
echo "Score with RNNsearch"
$SCORE --state search_state.pkl search_model.npz >search_scores.txt 2>>log.txt
check_result $NUMDIFF search_scores.txt "RNNsearch scores changed!"

echo "Stage 2: Test sampling"

SAMPLE="$GHOG/experiments/nmt/sample.py --source=english.txt --beam-search --beam-size 10"
echo "Sample with RNNencdec"
$SAMPLE --trans encdec_trans.txt --state encdec_state.pkl encdec_model.npz 2>>log.txt
check_result $DIFF encdec_trans.txt "RNNencdec translations changed!"
echo "Sample with RNNsearch"
THEANO_FLAGS="on_unused_input=warn" $SAMPLE --trans search_trans.txt --state search_state.pkl search_model.npz 2>>log.txt
check_result $DIFF search_trans.txt "RNNsearch translation changed!"

if [ $STATUS != "ok" ]
then
    echo "There are failed tests, check $DIR for the new outputs"    
fi  
