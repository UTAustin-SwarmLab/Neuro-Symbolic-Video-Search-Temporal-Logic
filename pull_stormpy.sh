#!/bin/bash
set -e

REPO_URL="https://github.com/moves-rwth/stormpy.git"
PREFIX="vendors/stormpy"
BRANCH="stable"

if [ -d "$PREFIX" ]; then
    echo "Subtree for stormpy already exists. Pulling updates..."
    git subtree pull --prefix="$PREFIX" "$REPO_URL" "$BRANCH" --squash
else
    echo "Subtree for stormpy does not exist. Adding subtree..."
    mkdir -p vendors
    git subtree add --prefix="$PREFIX" "$REPO_URL" "$BRANCH" --squash
fi

