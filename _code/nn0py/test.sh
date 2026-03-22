export PYTHONPATH=$PYTHONPATH:.

set -x
python test_gpt0.py ./_data/names.txt

