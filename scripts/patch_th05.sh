#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 SOURCE_KAIKI_DIR TH05PATCH_DIR OUTPUT_DIR" >&2
    exit 2
}

[[ $# == 3 ]] || usage
source_dir=$(realpath -- "$1")
patch_dir=$(realpath -- "$2")
output_dir=$(realpath -m -- "$3")
project_dir=$(realpath -- "$(dirname -- "$0")/..")

command -v bspatch >/dev/null || {
    echo "error: bspatch is required (Debian/Ubuntu package: bsdiff)" >&2
    exit 1
}
[[ -d $source_dir ]] || { echo "error: source directory not found: $source_dir" >&2; exit 1; }
[[ -d $patch_dir/patches/th05 ]] || { echo "error: th05patch tree not found: $patch_dir" >&2; exit 1; }
[[ ! -e $output_dir ]] || { echo "error: output already exists: $output_dir" >&2; exit 1; }

verify_hash() {
    local file=$1 expected=$2 actual
    [[ -f $file ]] || { echo "error: source file missing: $file" >&2; exit 1; }
    actual=$(sha256sum -- "$file")
    actual=${actual%% *}
    [[ $actual == "$expected" ]] || {
        echo "error: unsupported source hash for $file: $actual" >&2
        exit 1
    }
}

# The supplied old-five-games archive is a mixed revision: MAIN works with
# th05patch layout 1 while OP has the manifest hash for layout 2.
verify_hash "$source_dir/MAIN.EXE" c41f6e6b9a97b2433acc576ceaee800707c8d4ea7e498150a259663c8fa7d4f0
verify_hash "$source_dir/OP.EXE" c94efc071a4b1adf8ce6e2c5a7eefb9eb47c26f8bab61b2e58e736dda8130abb
verify_hash "$source_dir/ZUN.COM" 5044ae03f7c0333f37d88ef34e244502f90d7314287f6f387c07569e4e0b120e

mkdir -p -- "$output_dir"
cp -a -- "$source_dir"/. "$output_dir"/
bspatch "$source_dir/MAIN.EXE" "$output_dir/debloatm.exe" "$patch_dir/patches/th05/main/1"
bspatch "$source_dir/OP.EXE" "$output_dir/debloat.exe" "$patch_dir/patches/th05/op/2"
bspatch "$source_dir/ZUN.COM" "$output_dir/zum.com" "$patch_dir/patches/th05/zun/1"

mapfile -d '' data_one < <(find "$source_dir" -maxdepth 1 -type f -iname '*1.dat' -print0)
mapfile -d '' data_two < <(find "$source_dir" -maxdepth 1 -type f -iname '*2.dat' -print0)
[[ ${#data_one[@]} == 1 && ${#data_two[@]} == 1 ]] || {
    echo "error: expected exactly one *1.DAT and one *2.DAT" >&2
    exit 1
}
cp -- "${data_one[0]}" "$output_dir/kaiki_1.dat"
cp -- "${data_two[0]}" "$output_dir/kaiki_2.dat"

perl -pe 's/\bop\b/debloat/ig' "$source_dir/GAME.BAT" > "$output_dir/GAME.BAT.tmp"
mv -- "$output_dir/GAME.BAT.tmp" "$output_dir/GAME.BAT"
cp -- "$project_dir/config/th05_cpu/default.conf" "$output_dir/default.conf"
cp -- "$project_dir/config/th05_cpu/KAIKII.CFG" "$output_dir/KAIKII.CFG"

echo "patched: $output_dir"
