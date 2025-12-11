#!/bin/bash

IMAGES_DIR="./images"

# Create images folder
mkdir -p "$IMAGES_DIR"

echo "Moving all images to $IMAGES_DIR ..."
find . -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.gif" -o -iname "*.webp" \) \
    ! -path "./images/*" -exec mv -n {} "$IMAGES_DIR" \;

echo "Converting Obsidian image embeds..."

find . -type f -name "*.md" | while read -r file; do
    echo "Processing: $file"

    output="${file%.md}_converted.md"

    # macOS-safe sed commands with leading slash in src
    sed -E \
        -e 's/!\[\[([[:alnum:] _.-]+\.png)\]\]/<img src="\/images\/\1" alt="image" width="500">/g' \
        -e 's/!\[\[([[:alnum:] _.-]+\.jpg)\]\]/<img src="\/images\/\1" alt="image" width="500">/g' \
        -e 's/!\[\[([[:alnum:] _.-]+\.jpeg)\]\]/<img src="\/images\/\1" alt="image" width="500">/g' \
        -e 's/!\[\[([[:alnum:] _.-]+\.gif)\]\]/<img src="\/images\/\1" alt="image" width="500">/g' \
        -e 's/!\[\[([[:alnum:] _.-]+\.webp)\]\]/<img src="\/images\/\1" alt="image" width="500">/g' \
        "$file" > "$output"

done

echo "✔ Finished. Converted markdown saved as *_converted.md"

