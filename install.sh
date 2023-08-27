#!bin/bash
export role=$1

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# -- Downloading Required weights and data -- #
echo "Downloading GroundingDINO & Segment Anything Model (SAM) Weights"
WEIGHT_DIR=$ROOT_DIR/artifacts/weights
if [[ ! -e $WEIGHT_DIR ]]; then
    mkdir -p $WEIGHT_DIR
elif [[ ! -d $WEIGHT_DIR ]]; then
    echo "$WEIGHT_DIR already exists but is not a directory" 1>&2
fi

cd $WEIGHT_DIR
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# -- Installing GroundingDINO -- #
echo "Installing GroundingDINO"
git submodule init
git submodule update

cd $ROOT_DIR/video_to_automaton/GroundingDINO
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install torch
pip install -q -e .

# -- Installing video_to_automaton -- #
echo "Installing video_to_automaton"

python -m pip install --upgrade pip build

if [[ $role == "dev" ]];
then 
    python -m pip install --editable ."[dev, test]"
else
    python -m pip install -e .
fi
