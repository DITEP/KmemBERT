: '
Launch a bunch of scripts and check that everything runs smoothly

What we do not test here:
    - Resuming regression model and training in density mode and vice versa
    - History
'
TEST=0
PASSED=0

function checkError(){
    ((TEST++))
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "$(tput setaf 1)ERROR$(tput sgr0)"
    else
        ((PASSED++))
        echo -e "$(tput setaf 2)PASSED\n$(tput sgr0)"
        echo -e "\n----------------\n" >> test_output_tmp.txt
    fi
}

# Test baseline
echo "$(tput setaf 3)TESTING BASELINE...$(tput sgr0)"
python -m src.baseline -d ehr -mtf 1 > test_output_tmp.txt
checkError

# Test training and resume - density
echo "$(tput setaf 3)TESTING DENSITY TRAINING...$(tput sgr0)"
python -m src.training -d ehr -m density >> test_output_tmp.txt
checkError
RESUME_DENSE=$(ls -t results | head -1)
python -m src.training -d ehr -m density -r "${RESUME_DENSE}" >> test_output_tmp.txt
checkError

# Test training and resume - regression
echo "$(tput setaf 3)TESTING REGRESSION TRAINING...$(tput sgr0)"
python -m src.training -d ehr -m regression >> test_output_tmp.txt
checkError
RESUME_REG=$(ls -t results | head -1)
python -m src.training -d ehr -m regression -r "${RESUME_REG}" >> test_output_tmp.txt
checkError

# Test testing
echo "$(tput setaf 3)TESTING TESTING...$(tput sgr0)"
python -m src.testing -d ehr -r "${RESUME_REG}" >> test_output_tmp.txt
checkError

# Test hyperoptimization
echo "$(tput setaf 3)TESTING HYPEROPTIMIZATION...$(tput sgr0)"
python -m src.hyperoptimization -d ehr >> test_output_tmp.txt
checkError

# Test history
echo "$(tput setaf 3)TESTING HYPEROPTIMIZATION...$(tput sgr0)"
python -m src.hyperoptimization -d ehr >> test_output_tmp.txt
checkError

echo "$(tput setaf 3)TESTING FINISHED: ${PASSED}/${TEST} PASSED$(tput sgr0)"