#!/bin/sh

cmake ./build
echo "
-- Configuration done
"
cmake --build ./build
echo "
-- Build done
"
./build/Debug/sphere.exe

echo "
-- Executing done
"