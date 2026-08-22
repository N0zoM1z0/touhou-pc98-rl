#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 [--force] TEMPLATE.HDI OUTPUT.HDI [scenario options]" >&2
    echo "options: --stage N --phase N --end-phase N --character N --rank N --power N --lives N --bombs N" >&2
    exit 2
}

force=0
if [[ ${1:-} == --force ]]; then
    force=1
    shift
fi
[[ $# -ge 2 ]] || usage

template=$(realpath -- "$1")
output=$(realpath -m -- "$2")
shift 2

stage=1
phase=0
end_phase=0
character=2
rank=0
power=0
lives=3
bombs=3

while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || usage
    case $1 in
        --stage) stage=$2 ;;
        --phase) phase=$2 ;;
        --end-phase) end_phase=$2 ;;
        --character) character=$2 ;;
        --rank) rank=$2 ;;
        --power) power=$2 ;;
        --lives) lives=$2 ;;
        --bombs) bombs=$2 ;;
        *) usage ;;
    esac
    shift 2
done

require_range() {
    local name=$1 value=$2 minimum=$3 maximum=$4
    [[ $value =~ ^[0-9]+$ ]] || {
        echo "error: $name must be an integer" >&2
        exit 1
    }
    (( value >= minimum && value <= maximum )) || {
        echo "error: $name must be in [$minimum, $maximum]" >&2
        exit 1
    }
}

require_range stage "$stage" 0 6
require_range phase "$phase" 0 12
require_range end-phase "$end_phase" 0 12
require_range character "$character" 0 3
require_range rank "$rank" 0 3
require_range power "$power" 0 128
require_range lives "$lives" 0 3
require_range bombs "$bombs" 0 3

command -v mcopy >/dev/null || {
    echo "error: mcopy is required (Debian/Ubuntu package: mtools)" >&2
    exit 1
}
[[ -f $template ]] || { echo "error: template HDI not found: $template" >&2; exit 1; }
[[ $template != "$output" ]] || {
    echo "error: refusing to overwrite the template HDI" >&2
    exit 1
}
if [[ -e $output && $force == 0 ]]; then
    echo "error: output exists; pass --force to replace it: $output" >&2
    exit 1
fi

temporary=$(mktemp -d)
trap 'rm -rf -- "$temporary"' EXIT
printf '%s\n' \
    "total_live=$lives" \
    "total_bomb=$bombs" \
    "skip_to=$stage" \
    "skip_to_boss_phase=$phase" \
    "end_phase=$end_phase" \
    "char=$character" \
    "rank=$rank" \
    "power=$power" > "$temporary/KAIKII.CFG"

mkdir -p -- "$(dirname -- "$output")"
cp --reflink=auto -- "$template" "$output"
MTOOLS_SKIP_CHECK=1 mcopy -o -i "${output}@@38912" \
    "$temporary/KAIKII.CFG" ::KAIKI/KAIKII.CFG

echo "prepared scenario: $output"
echo "stage=$stage phase=$phase end_phase=$end_phase character=$character rank=$rank power=$power lives=$lives bombs=$bombs"

