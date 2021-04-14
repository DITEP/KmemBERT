# Launch a bunch of scripts and check that everything runs smoothly

function checkError(){
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "ERROR"
        exit
    else
        echo "PASSED"
    fi
}

python -m src.baseline -d ehr -mtf 1 > test_output_tmp.txt
checkError
python -m src.training -d ehr >> test_output_tmp.txt
checkError