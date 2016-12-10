#!/bin/sh
echo
if [ -a .commit ]
    then
    # remove temporary file
    rm .commit
    echo $(pwd)
    # add potential changes to the docs
    git add docs
    # combine these canges with the previous commit
    git commit --amend -C HEAD --no-verify
fi
exit 0
