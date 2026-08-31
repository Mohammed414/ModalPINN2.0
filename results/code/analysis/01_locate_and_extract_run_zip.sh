#!/bin/bash
# Locate and extract the matched-effort re-run archive downloaded from Colab.
set -e

cd /Users/mohammedhilal/Desktop/try && ls -lat | head -20
echo "=== zips anywhere under try (recent first) ==="
find /Users/mohammedhilal/Desktop/try -maxdepth 3 -iname "*.zip" -exec ls -lat {} + 2>/dev/null | head -20
echo "=== any _WARM or matched dirs ==="
find /Users/mohammedhilal/Desktop/try -maxdepth 5 -iname "*matched*" -o -maxdepth 5 -iname "*WARM*" 2>/dev/null | head -20

# Extract the located archive into a scratch workspace directory (zx/).
Z="/Users/mohammedhilal/Desktop/try/ModalPINN2.0/modes_experiment/notebooks/matched_effort/baseline_physics_only_K3_matched-20260828T092457Z-1-001.zip"
rm -rf zx && mkdir -p zx && unzip -q "$Z" -d zx
echo "=== tree (2 levels) ==="
find zx -maxdepth 3 | sort | head -40
du -sh zx/* 2>/dev/null
