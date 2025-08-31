#!/bin/bash

echo "Downloading MNIST dataset..."

# Function to download and check file
download_file() {
    local url=$1
    local filename=$2
    local output_file=$3
    
    if [ ! -f "$output_file" ]; then
        echo "Downloading $filename..."
        if curl -L -o "$filename" "$url"; then
            if file "$filename" | grep -q "gzip"; then
                gunzip "$filename"
                echo "✓ Downloaded and extracted $output_file"
            else
                echo "✗ Download failed for $filename - file is not gzip format"
                rm -f "$filename"
                return 1
            fi
        else
            echo "✗ Failed to download $filename"
            return 1
        fi
    else
        echo "✓ $output_file already exists"
    fi
}

# Try primary URLs first, then fallback URLs
BASE_URL="http://yann.lecun.com/exdb/mnist"
FALLBACK_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

# Download files
download_file "$BASE_URL/train-images-idx3-ubyte.gz" "train-images-idx3-ubyte.gz" "train-images-idx3-ubyte" || \
download_file "$FALLBACK_URL/train-images-idx3-ubyte.gz" "train-images-idx3-ubyte.gz" "train-images-idx3-ubyte"

download_file "$BASE_URL/train-labels-idx1-ubyte.gz" "train-labels-idx1-ubyte.gz" "train-labels-idx1-ubyte" || \
download_file "$FALLBACK_URL/train-labels-idx1-ubyte.gz" "train-labels-idx1-ubyte.gz" "train-labels-idx1-ubyte"

download_file "$BASE_URL/t10k-images-idx3-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-images-idx3-ubyte" || \
download_file "$FALLBACK_URL/t10k-images-idx3-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-images-idx3-ubyte"

download_file "$BASE_URL/t10k-labels-idx1-ubyte.gz" "t10k-labels-idx1-ubyte.gz" "t10k-labels-idx1-ubyte" || \
download_file "$FALLBACK_URL/t10k-labels-idx1-ubyte.gz" "t10k-labels-idx1-ubyte.gz" "t10k-labels-idx1-ubyte"

echo ""
echo "MNIST download complete!"
echo "Available files:"
ls -la *-ubyte 2>/dev/null || {
    echo "No MNIST files found. Manual download required:"
    echo "Visit: http://yann.lecun.com/exdb/mnist/"
    echo "Download all 4 .gz files and extract them here."
    exit 1
}