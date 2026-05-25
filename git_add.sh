#!/bin/bash

rm -f .git/index.lock

BASE_DIR="TEXTBOOKS (PUBLIC)"

# Root files
root_items=(
  "README.md"
  "move_images.sh"
  "Table of Contents.base"
)

for item in "${root_items[@]}"
do
  echo ""
  echo "Adding: $item"

  git add "$item"

  if git diff --cached --quiet; then
    echo "Nothing to commit for $item"
    continue
  fi

  git commit -m "Add $item"
  git push

  if [ $? -ne 0 ]; then
    echo "Push failed on $item"
    exit 1
  fi
done

# Textbook folders
for category in "$BASE_DIR"/*
do
  if [ -d "$category" ]; then

    for subfolder in "$category"/*
    do
      if [ -d "$subfolder" ]; then

        echo ""
        echo "Adding: $subfolder"

        git add "$subfolder"

        if git diff --cached --quiet; then
          echo "Nothing staged for $subfolder"
          continue
        fi

        folder_name=$(basename "$subfolder")

        git commit -m "Add $folder_name"
        git push

        if [ $? -ne 0 ]; then
          echo "Push failed on $subfolder"
          exit 1
        fi
      fi
    done
  fi
done

# Loose markdown files
git add "$BASE_DIR"/*.md 2>/dev/null

if ! git diff --cached --quiet; then
  git commit -m "Add textbook notes"
  git push
fi

echo ""
echo "Finished successfully!"
git status
