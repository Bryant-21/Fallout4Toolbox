import os
import struct

RIFF_HEADER_SIZE = 12
CHUNK_HEADER_SIZE = 8

# Chunks we want to preserve
KEEP_CHUNKS = {
    b"fmt ",
    b"data",
    b"smpl",
    b"cue ",
    b"LIST",   # we will filter LIST type later
}

def read_chunk(f):
    header = f.read(CHUNK_HEADER_SIZE)
    if len(header) < CHUNK_HEADER_SIZE:
        return None, None

    chunk_id, size = struct.unpack("<4sI", header)
    data = f.read(size)

    # Chunks are padded to even size
    if size % 2 == 1:
        f.read(1)

    return chunk_id, data


def write_chunk(f, chunk_id, data):
    f.write(struct.pack("<4sI", chunk_id, len(data)))
    f.write(data)
    if len(data) % 2 == 1:
        f.write(b"\x00")


def should_keep_list_chunk(data):
    # LIST chunk begins with subtype like "INFO" or "adtl"
    if len(data) < 4:
        return False
    list_type = data[:4]
    return list_type == b"adtl"


def strip_metadata(input_path, output_path):
    with open(input_path, "rb") as f:
        riff, size, wave = struct.unpack("<4sI4s", f.read(RIFF_HEADER_SIZE))
        if riff != b"RIFF" or wave != b"WAVE":
            raise ValueError("Not a WAV file")

        chunks_to_write = []

        while True:
            chunk_id, data = read_chunk(f)
            if chunk_id is None:
                break

            if chunk_id == b"LIST":
                if should_keep_list_chunk(data):
                    chunks_to_write.append((chunk_id, data))
            elif chunk_id in KEEP_CHUNKS:
                chunks_to_write.append((chunk_id, data))
            # else: drop metadata

    # Write new RIFF
    with open(output_path, "wb") as out:
        out.write(b"RIFF")
        out.write(b"\x00\x00\x00\x00")  # placeholder size
        out.write(b"WAVE")

        start = out.tell()

        for cid, data in chunks_to_write:
            write_chunk(out, cid, data)

        end = out.tell()

        # Fix RIFF size
        out.seek(4)
        out.write(struct.pack("<I", end - 8))


def process_file(path):
    base, ext = os.path.splitext(path)
    output = f"{base}{ext}"
    strip_metadata(path, output)
    print("Cleaned:", output)


if __name__ == "__main__":

    target = r"I:\Fallout4Mods\B21_PlasmaCaster\Data\Sound\FX\WPN\B21_PlasmaCaster"
    if os.path.isdir(target):
        for root, _, files in os.walk(target):
            for f in files:
                if f.lower().endswith(".wav"):
                    process_file(os.path.join(root, f))
