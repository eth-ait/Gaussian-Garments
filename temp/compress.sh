#!/bin/bash
# Usage: ./script.sh <directory_path> [archive_name]
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <directory_path> [archive_name]"
    exit 1
fi

SOURCE_DIR="$1"
ARCHIVE_NAME="${2:-filtered_archive.tar.gz}"

# Change directory into SOURCE_DIR so that globbing works as intended.
# The archive will be created in the original directory.
(
  cd "$SOURCE_DIR" || { echo "Cannot cd to $SOURCE_DIR"; exit 1; }
  tar -cf "$ARCHIVE_NAME" \
    cameras.json \
    Take*/cameras.json \
    Take*/*/capture_images \
    Take*/*/capture_labels \
    Take*/Meshes/smplx
)

echo "Archive created: $ARCHIVE_NAME"
