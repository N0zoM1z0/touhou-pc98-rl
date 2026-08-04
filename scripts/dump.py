import argparse
import json
import time
from pathlib import Path

import rrr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump game map and raw entity state"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        required=True,
        help="Interval between frames in ms",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output dir",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["colon", "raw"],
        required=True,
        help="Format: 'colon' or 'raw'",
    )
    parser.add_argument(
        "-s",
        "--separate",
        type=str,
        choices=["y", "n"],
        required=True,
        help="Separate files per frame (y) or all in one file (n)",
    )
    return parser.parse_args()


def to_colon(data: bytes) -> str:
    return ":".join(f"{b:02X}" for b in data)


def serialize_colon(frame: rrr.RawFrame) -> str:
    state_bytes = frame.__getstate__()
    return to_colon(state_bytes)


def serialize_raw(frame: rrr.RawFrame) -> bytes:
    return frame.__getstate__()


def dump(watcher: rrr.MemoryWatcher, output_dir: Path,
                    interval_ms: int, format_type: str, separate: bool):
    """Continuously dump frames until the game ends.
    :param: rrr's memory watcher
    :param: directory to output
    :param: The interval between frames recorded
    :param: if separated dumped data
    :raises: KeyboardInterrupt if ctrl-c is pressed.
    """
    interval_sec = interval_ms / 1000.0
    frames = []
    frame_count = 0
    last_frame_data = None
    print(f"Starting dump (interval={interval_ms}ms, format={format_type}, separate={separate})")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            start_time = time.time()
            
            result = watcher.read_state()
            if result is None:
                time.sleep(interval_sec)
                continue
            
            features, maps, game_end_flag, rewards, raw_frame = result
            current_frame_data = serialize_raw(raw_frame)
            if current_frame_data == last_frame_data:
                time.sleep(max(0, interval_sec - (time.time() - start_time)))
                continue
            
            last_frame_data = current_frame_data
            frames.append(raw_frame)
            frame_count += 1
            
            if separate:
                frame_filename = output_dir / f"frame_{frame_count:06d}"
                if format_type == "colon":
                    frame_filename = frame_filename.with_suffix(".txt")
                    with open(frame_filename, "w") as f:
                        f.write(serialize_colon(raw_frame))
                else:  # raw
                    frame_filename = frame_filename.with_suffix(".bin")
                    with open(frame_filename, "wb") as f:
                        f.write(current_frame_data)
            
            # Print progress every 30
            if frame_count % 30 == 0:
                print(f"Dumped {frame_count} frames")
            
            # Check for game end
            if game_end_flag != 0:
                print(f"Game ended (flag={game_end_flag})")
                break

            elapsed = time.time() - start_time
            sleep_time = max(0, interval_sec - elapsed)
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    if not separate and frames:
        if format_type == "colon":
            combined_file = output_dir / "all_frames.txt"
            with open(combined_file, "w") as f:
                for i, frame in enumerate(frames):
                    f.write(f"# Frame {i}\n")
                    f.write(serialize_colon(frame) + "\n")
            print(f"Saved all frames to {combined_file}")
        else:  # raw
            combined_file = output_dir / "all_frames.bin"
            with open(combined_file, "wb") as f:
                for frame in frames:
                    f.write(serialize_raw(frame))
            print(f"Saved all frames to {combined_file}")

    metadata = {
        "frame_count": frame_count,
        "interval_ms": interval_ms,
        "format": format_type,
        "separate_files": separate,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")

    if frames:
        print(f"\nReconstructing maps from {frame_count} frames...")
        try:
            maps_bytes = rrr.rec_maps_bytes(frames)
            maps_file = output_dir / "maps.bin"
            with open(maps_file, "wb") as f:
                f.write(maps_bytes)
            print(f"Saved reconstructed maps to {maps_file}")
            print(f"Map data size: {len(maps_bytes) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"Warning: Failed to reconstruct maps: {e}. Not enough space?")


def main():
    args = parse_args()
    
    # mkdir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    watcher = rrr.MemoryWatcher()

    separate = args.separate.lower() == "y"
    dump(
        watcher,
        output_dir,
        args.interval,
        args.format,
        separate,
    )
    
    print("Dump completed!")


if __name__ == "__main__":
    main()

