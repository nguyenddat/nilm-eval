import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(entry, destination):
    if destination.is_file() and checksum(destination) == entry["sha256"]:
        print(f"Đã có tệp hợp lệ: {destination}", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, 4):
        try:
            print(f"Đang tải {entry['url']} (lần {attempt}/3)...", flush=True)
            # User-Agent mặc định của urllib bị máy chủ nguồn từ chối.
            request = Request(entry["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            assert checksum(partial) == entry["sha256"], f"SHA-256 không khớp: {partial}. Kiểm tra kết nối hoặc phiên bản dữ liệu nguồn."
            partial.replace(destination)
            return
        except (OSError, URLError, AssertionError) as error:
            print(f"Tải tệp thất bại: {error}", file=sys.stderr, flush=True)
            if attempt == 3:
                raise
            time.sleep(2)


def run_7zip(executable, *arguments):
    result = subprocess.run([executable, *map(str, arguments)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, f"7-Zip xử lý thất bại:\n{result.stdout}\n{result.stderr}"
    return result.stdout


def archive_files(executable, archive, archive_root):
    listing = run_7zip(executable, "l", "-slt", "-sccUTF-8", archive)
    assert "----------\n" in listing, f"Không đọc được danh sách tệp trong {archive}."
    entries = listing.split("----------\n", 1)[1]
    files = {}
    for block in entries.strip().split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if not fields or "D" in fields.get("Attributes", "") or fields.get("Folder") == "+":
            continue
        relative = Path(fields["Path"])
        assert not relative.is_absolute() and ".." not in relative.parts, f"Đường dẫn trong archive không hợp lệ: {relative}."
        if archive_root:
            relative = relative.relative_to(archive_root)
        files[relative] = int(fields["Size"])
    assert files, f"Archive không chứa dữ liệu: {archive}."
    return files


def files_complete(directory, files):
    return all((directory / path).is_file() and (directory / path).stat().st_size == size for path, size in files.items())


def extract(executable, archive, destination, archive_root):
    files = archive_files(executable, archive, archive_root)
    if files_complete(destination, files):
        print(f"Đã có đủ {len(files)} tệp đúng kích thước: {destination}", flush=True)
        return

    print(f"Đang giải nén {archive} vào {destination}...", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Chỉ đưa dữ liệu vào thư mục đích sau khi giải nén hoàn tất và kiểm tra đủ tệp.
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        run_7zip(executable, "x", "-y", "-sccUTF-8", f"-o{staging}", archive)
        source = staging / archive_root
        assert files_complete(source, files), f"Dữ liệu giải nén chưa đầy đủ: {archive}."
        for relative in files:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            (source / relative).replace(target)


def main():
    parser = argparse.ArgumentParser(description="Tải REFIT và PLEGMA, kiểm tra SHA-256 và giải nén bản cleaned vào đúng thư mục của dự án.")
    parser.add_argument("--dataset", choices=["all", "refit", "plegma"], default="all", help="Bộ dữ liệu cần tải; mặc định tải cả hai.")
    parser.add_argument("--include-raw", action="store_true", help="Tải thêm archive raw gốc và README đi kèm; không giải nén bản raw.")
    parser.add_argument("--download-only", action="store_true", help="Chỉ tải các tệp nguồn, không giải nén.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Thư mục data đích; mặc định lấy theo vị trí script, không phụ thuộc thư mục chạy lệnh.")
    args = parser.parse_args()

    executable = shutil.which("7zz") or shutil.which("7z")
    assert args.download_only or executable, "Cần cài 7-Zip (lệnh 7z hoặc 7zz). Trên Ubuntu/Debian: sudo apt install p7zip-full."
    manifest = json.loads((SCRIPT_DIR / "datasets.json").read_text(encoding="utf-8"))
    selected = list(manifest) if args.dataset == "all" else [args.dataset.upper()]
    for dataset in selected:
        for entry in manifest[dataset]:
            if entry["include_raw"] and not args.include_raw:
                continue
            archive = args.data_dir / "raw" / dataset / entry["filename"]
            download(entry, archive)
            if entry["extract"] and not args.download_only:
                extract(executable, archive, args.data_dir / "cleaned" / dataset, entry["archive_root"])
    print("Đã hoàn tất chuẩn bị dữ liệu.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, URLError, AssertionError, ValueError) as error:
        print(f"Không thể chuẩn bị dữ liệu: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nĐã dừng tải dữ liệu. Chạy lại lệnh để thử lại các tệp chưa hoàn tất.", file=sys.stderr)
        sys.exit(130)
