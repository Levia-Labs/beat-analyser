"""Process and inspect audio projects without the web server.

Usage:
  python app/cli.py FILE [FILE ...]      analyse local audio files
  python app/cli.py --youtube URL        download and analyse a YouTube video
  python app/cli.py --summary NAME       rows and time range per track of one project
  python app/cli.py --list               every project and whether its ZIP exists
  python app/cli.py --repair             rebuild CSVs and ZIP of every project
  add --root DIR to any of them (default: uploads)

Writes the same project folders and ZIPs the server does, so the web UI lists them. Output
is short and line-oriented so an agent can read it without opening the CSVs.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import analysis
import pipeline
import storage
import youtube

YT_MAX_SECONDS = 20 * 60
YT_MAX_BYTES = 200 * 1024 * 1024


def analyse_files(root: Path, files: list[Path]) -> int:
    failed = 0
    for source in files:
        try:
            if not source.is_file():
                raise FileNotFoundError(f"no such file: {source}")
            name = storage.project_name(source.name)

            def place_audio(wav: Path, source: Path = source) -> None:
                if source.suffix.lower() == ".wav":
                    shutil.copyfile(source, wav)
                else:
                    analysis.to_wav(source, wav)

            pipeline.build(root, name, place_audio)
            summary(root, name)
        except Exception as exc:
            failed += 1
            print(f"{source}: {exc}", file=sys.stderr)
    return 1 if failed else 0


def analyse_youtube(root: Path, url: str) -> int:
    vid = youtube.video_id(url)

    def place_audio(wav: Path) -> None:
        source = youtube.download_audio(vid, wav.parent, YT_MAX_SECONDS, YT_MAX_BYTES)
        try:
            analysis.to_wav(source, wav)
        finally:
            source.unlink(missing_ok=True)

    pipeline.build(root, vid, place_audio)
    return summary(root, vid)


def summary(root: Path, name: str) -> int:
    folder = storage.project_dir(root, name)
    if not storage.is_valid_name(name) or not folder.is_dir():
        print(f"no project '{name}' under {root}", file=sys.stderr)
        return 1
    archive = storage.zip_path(root, name)
    print(f"{name}  zip={'yes' if archive.is_file() else 'no'}  folder={folder}")
    for track in sorted(folder.glob("*.csv")):
        with track.open(newline="", encoding="utf-8") as handle:
            times = [float(row[0]) for row in csv.reader(handle) if row]
        span = f"{times[0]:.3f}s..{times[-1]:.3f}s" if times else "-"
        print(f"  {track.stem:<8} {len(times):>6} events  {span}")
    return 0


def list_projects(root: Path) -> int:
    projects = storage.list_projects(root)
    if not projects:
        print(f"no projects under {root}")
    for name, files in projects.items():
        has_zip = storage.zip_path(root, name).is_file()
        print(f"  {name:<40} files={len(files):<3} zip={'yes' if has_zip else 'no'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("files", nargs="*", type=Path, help="audio files to analyse")
    parser.add_argument("--root", type=Path, default=Path("uploads"), help="upload root")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--youtube", metavar="URL", help="download and analyse a video")
    group.add_argument("--summary", metavar="NAME", help="summarise one project")
    group.add_argument("--list", action="store_true", help="list projects")
    group.add_argument("--repair", action="store_true", help="rebuild every project")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    if args.repair:
        for row in pipeline.repair_all(args.root):
            print(row)
        return 0
    if args.list:
        return list_projects(args.root)
    if args.summary:
        return summary(args.root, args.summary)
    if args.youtube:
        try:
            return analyse_youtube(args.root, args.youtube)
        except Exception as exc:
            print(f"{args.youtube}: {exc}", file=sys.stderr)
            return 1
    if not args.files:
        parser.error("name at least one file, or one of --youtube/--summary/--list/--repair")
    return analyse_files(args.root, args.files)


if __name__ == "__main__":
    sys.exit(main())
