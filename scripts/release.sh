#!/usr/bin/env bash

# Requirements
# brew install hub
# npm install -g git-release-notes
# pip install twine wheel

set -ex

PYPROJECT_FILE='pyproject.toml'
VER_TAG='__version__ = '
RELEASENOTE='./node_modules/.bin/git-release-notes'

function escape_slashes {
    sed 's/\//\\\//g'
}

function update_ver_line {
    local OLD_LINE_PATTERN=$1
    local NEW_LINE=$2
    local FILE=$3

    local NEW=$(echo "${NEW_LINE}" | escape_slashes)
    sed -i '/'"${OLD_LINE_PATTERN}"'/s/.*/'"${NEW}"'/' "${FILE}"
    head -n10 ${FILE}
}

BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ "$BRANCH" != "main" ]]; then
  printf "You are not at main branch, exit\n";
  exit 1;
fi

LAST_UPDATE=`git show --no-notes --format=format:"%H" $BRANCH | head -n 1`
LAST_COMMIT=`git show --no-notes --format=format:"%H" origin/$BRANCH | head -n 1`

if [ $LAST_COMMIT != $LAST_UPDATE ]; then
    printf "Your local $BRANCH is behind the remote master, exit\n"
    exit 1;
fi

# release the current version
export RELEASE_VER=$(poetry version --short)
LAST_VER=$(git tag -l | sort -V | tail -n1)
printf "last version: \e[1;32m$LAST_VER\e[0m\n"

if [[ $1 == "final" ]]; then
  printf "this will be a final release: \e[1;33m$RELEASE_VER\e[0m\n"

  NEXT_VER=$(echo $RELEASE_VER | awk -F. -v OFS=. 'NF==1{print ++$NF}; NF>1{$NF=sprintf("%0*d", length($NF), ($NF+1)); print}')
  printf "bump master version to: \e[1;32m$NEXT_VER\e[0m\n"

#  make_release_note
#
#  pub_pypi client
#  pub_pypi server
#  cp scripts/MANIFEST.in ./
#  cp scripts/setup.py ./
#  pub_pypi "."

#  VER_TAG_NEXT=$VER_TAG\'${NEXT_VER}\'
#  update_ver_line "$VER_TAG" "$VER_TAG_NEXT" 'client/clip_client/__init__.py'
#  update_ver_line "$VER_TAG" "$VER_TAG_NEXT" 'server/clip_server/__init__.py'
#  RELEASE_REASON="$2"
#  RELEASE_ACTOR="$3"
#  git_commit
elif [[ $1 == 'rc' ]]; then
  printf "this will be a release candidate: \e[1;33m$RELEASE_VER\e[0m\n"
  DOT_RELEASE_VER=$(echo $RELEASE_VER | sed "s/rc/\./")
  NEXT_VER=$(echo $DOT_RELEASE_VER | awk -F. -v OFS=. 'NF==1{print ++$NF}; NF>1{$NF=sprintf("%0*d", length($NF), ($NF+1)); print}')
  NEXT_VER=$(echo $NEXT_VER | sed "s/\.\([^.]*\)$/rc\1/")
  printf "bump master version to: \e[1;32m$NEXT_VER\e[0m, this will be the next version\n"

#  make_release_note
#
#  pub_pypi client
#  pub_pypi server
#  cp scripts/MANIFEST.in ./
#  cp scripts/setup.py ./
#  pub_pypi "."
#
#  VER_TAG_NEXT=$VER_TAG\'${NEXT_VER}\'
#  update_ver_line "$VER_TAG" "$VER_TAG_NEXT" 'client/clip_client/__init__.py'
#  update_ver_line "$VER_TAG" "$VER_TAG_NEXT" 'server/clip_server/__init__.py'
#  RELEASE_REASON="$2"
#  RELEASE_ACTOR="$3"
#  git_commit
else
  # as a prerelease, pypi update only, no back commit etc.
  COMMITS_SINCE_LAST_VER=$(git rev-list $LAST_VER..HEAD --count)
  NEXT_VER=$RELEASE_VER".rc"$COMMITS_SINCE_LAST_VER
  printf "this will be a developmental release: \e[1;33m$NEXT_VER\e[0m\n"

#  VER_TAG_NEXT=$VER_TAG\'${NEXT_VER}\'
#  update_ver_line "$VER_TAG" "$VER_TAG_NEXT" 'client/clip_client/__init__.py'
#  update_ver_line "$VER_TAG" "$VER_TAG_NEXT" 'server/clip_server/__init__.py'
#
#  pub_pypi client
#  pub_pypi server
#  cp scripts/MANIFEST.in ./
#  cp scripts/setup.py ./
#  pub_pypi "."
fi
