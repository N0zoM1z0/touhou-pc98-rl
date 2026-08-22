#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 [--force] SOURCE.HDI PATCHED_DIR OUTPUT.HDI" >&2
    exit 2
}

force=0
if [[ ${1:-} == --force ]]; then
    force=1
    shift
fi
[[ $# == 3 ]] || usage

source_hdi=$(realpath -- "$1")
patched_dir=$(realpath -- "$2")
output_hdi=$(realpath -m -- "$3")
partition_offset=38912

command -v mcopy >/dev/null || {
    echo "error: mcopy is required (Debian/Ubuntu package: mtools)" >&2
    exit 1
}
[[ -f $source_hdi ]] || { echo "error: source HDI not found: $source_hdi" >&2; exit 1; }
[[ -d $patched_dir ]] || { echo "error: patched directory not found: $patched_dir" >&2; exit 1; }
[[ $source_hdi != "$output_hdi" ]] || {
    echo "error: refusing to overwrite the source HDI" >&2
    exit 1
}
if [[ -e $output_hdi && $force == 0 ]]; then
    echo "error: output exists; pass --force to replace it: $output_hdi" >&2
    exit 1
fi

required=(debloatm.exe debloat.exe zum.com GAME.BAT KAIKII.CFG kaiki_1.dat kaiki_2.dat)
for file in "${required[@]}"; do
    [[ -f $patched_dir/$file ]] || {
        echo "error: patched file missing: $patched_dir/$file" >&2
        exit 1
    }
done

mkdir -p -- "$(dirname -- "$output_hdi")"
cp --reflink=auto -- "$source_hdi" "$output_hdi"
image="${output_hdi}@@${partition_offset}"

copy_into_game() {
    local source=$1 destination=$2
    MTOOLS_SKIP_CHECK=1 mcopy -o -i "$image" "$patched_dir/$source" "::KAIKI/$destination"
}

copy_into_game debloatm.exe DEBLOATM.EXE
copy_into_game debloat.exe DEBLOAT.EXE
copy_into_game zum.com ZUM.COM
copy_into_game GAME.BAT GAME.BAT
copy_into_game KAIKII.CFG KAIKII.CFG
copy_into_game kaiki_1.dat KAIKI_1.DAT
copy_into_game kaiki_2.dat KAIKI_2.DAT

echo "prepared: $output_hdi"
