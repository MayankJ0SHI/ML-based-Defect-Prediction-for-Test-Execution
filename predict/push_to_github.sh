#!/bin/bash

# push_to_github.sh
# Use command-line arguments for Git credentials
GH_USERNAME=$1
GH_PAT=$2
REPO=https://${GH_USERNAME}:${GH_PAT}@<GIT-REPO>
cd /model/predict

echo "Executing push_to_github.sh script at $(date)" >> /usr/src/app/script_log.txt

# Configure Git user information
git config user.email "<EMAIL-ID>"
git config user.name "<USERNAME>"

# Check if there are changes
if git diff-index --quiet HEAD --; then
  echo "No changes to push."
else
  # Add changes to the staging area
git add TextModel.pkl

# Add all changes including deletions and untracked files
git add -A

# Commit changes
git commit -m "Update TextModel.pkl"


  # Set up Git credentials using command-line arguments
  git remote set-url origin ${REPO}

  # Get the current branch name
  current_branch=$(git symbolic-ref --short HEAD 2>/dev/null)

  # Check if "Defect_Classification_Model_History" exists remotely
  if git ls-remote --exit-code --heads ${REPO} "refs/heads/Defect_Classification_Model_History"; then
    # Branch exists, switch to it
    git fetch origin Defect_Classification_Model_History:Defect_Classification_Model_History
    git checkout Defect_Classification_Model_History
  else
    # Branch doesn't exist, create a new branch and switch to it
    git checkout -b Defect_Classification_Model_History
  fi

  # Push changes to the branch
  git push origin Defect_Classification_Model_History
fi

echo "Script execution complete at $(date)" >> /usr/src/app/script_log.txt
