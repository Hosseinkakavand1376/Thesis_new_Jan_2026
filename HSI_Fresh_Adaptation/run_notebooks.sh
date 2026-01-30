#!/bin/zsh
# Run all Jupyter notebooks with '10' in the filename in the current directory
for nb in *10*.ipynb; do
	if [[ -f "$nb" ]]; then
		echo "Running $nb"
        papermill "$nb" "$nb"
	fi
done
