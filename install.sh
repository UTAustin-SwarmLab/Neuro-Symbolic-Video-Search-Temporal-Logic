#!bin/bash
export role=$2

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#-- Downloading Required weights and data -- #
echo "Downloading Yolo V8"
WEIGHT_DIR=$ROOT_DIR/artifacts/weights
if [[ ! -e $WEIGHT_DIR ]]; then
    mkdir -p $WEIGHT_DIR
    cd $WEIGHT_DIR
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
elif [[ ! -d $WEIGHT_DIR ]]; then
    echo "$WEIGHT_DIR already exists but is not a directory" 1>&2
fi

# -- Installing video_to_automaton -- #
echo "Installing video_to_automaton"
cd $ROOT_DIR
python3 -m pip install --upgrade pip build
if [[ $role == "dev" ]];
then 
    python3 -m pip install --editable ."[dev, test]"
else
    python3 -m pip install -e .
fi
