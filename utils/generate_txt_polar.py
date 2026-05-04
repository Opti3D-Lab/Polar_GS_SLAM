import os
import glob
import sys

def find_normal_dir(base_dir):
    """Find directory containing 'normal' in its name"""
    for dir_name in os.listdir(base_dir):
        if 'normal' in dir_name.lower() and os.path.isdir(os.path.join(base_dir, dir_name)):
            return dir_name
    return None


def get_files_with_timestamps(directory):
    """Get timestamps and filenames from a directory"""
    files_with_timestamps = []
    for filename in os.listdir(directory):
        # Extract the numeric part before any underscore or dot
        parts = filename.split('_')[0].split('.')
        if parts[0].isdigit():
            # Divide by 1e3 instead of 1e6 and format with three extra zeros
            timestamp = float(parts[0]) / 1e3
            # Format the timestamp to ensure it has exactly 3 decimal places
            timestamp = float(f"{timestamp:.3f}000")
            full_path = os.path.join(directory, filename)
            files_with_timestamps.append((timestamp, filename))
    return sorted(files_with_timestamps)


def compare_timestamps(list1, list2, dir1_name, dir2_name):
    """Compare timestamps between two lists and print warnings if they don't match"""
    timestamps1 = set(t for t, _ in list1)
    timestamps2 = set(t for t, _ in list2)

    if timestamps1 != timestamps2:
        print(f"\nWarning: Timestamps don't match between {dir1_name} and {dir2_name}")
        only_in_1 = timestamps1 - timestamps2
        only_in_2 = timestamps2 - timestamps1
        if only_in_1:
            print(f"Timestamps only in {dir1_name}: {sorted(only_in_1)}")
        if only_in_2:
            print(f"Timestamps only in {dir2_name}: {sorted(only_in_2)}")
        return False
    return True


def write_timestamp_file(files_with_timestamps, directory, output_file):
    """Write timestamps and relative paths to output file"""
    with open(output_file, 'w') as f:
        for timestamp, filename in files_with_timestamps:
            relative_path = os.path.join(directory, filename)
            f.write(f"{timestamp:.6f} {relative_path}\n")


def main():
    # Base directory
    base_directory = r"/data/datasets/my/polar_depth_V2/water_dispenser_620_around_xyz/"

    # Define subdirectories
    rgb_dir = "Id_enhance"
    aolp_dir = "aolp_1chanel"
    dolp_dir = "dolp_1chanel"
    depth_dir = "depth_pro"
    normal_dir = "depth_pro_normal"    # Find directory containing 'normal'
    seg_dir = "polar_seg_dolp"
    print("注意默认rgb文件夹是Id_final_wb，seg文件夹默认polar_seg_dolp")

    # Get full paths
    rgb_path = os.path.join(base_directory, rgb_dir)
    aolp_path = os.path.join(base_directory, aolp_dir)
    dolp_path = os.path.join(base_directory, dolp_dir)
    depth_path = os.path.join(base_directory, depth_dir)
    normal_path = os.path.join(base_directory, normal_dir)
    seg_path = os.path.join(base_directory, seg_dir)

    # Get files with timestamps for each directory
    rgb_files = get_files_with_timestamps(rgb_path)
    aolp_files = get_files_with_timestamps(aolp_path)
    dolp_files = get_files_with_timestamps(dolp_path)
    depth_files = get_files_with_timestamps(depth_path) if os.path.exists(depth_path) else []
    normal_files = get_files_with_timestamps(normal_path) if os.path.exists(normal_path) else []
    seg_files = get_files_with_timestamps(seg_path) if os.path.exists(seg_path) else []

    # Compare timestamps
    print("\nValidating timestamps...")
    rgb_valid = True
    depth_valid = True
    seg_full = True

    # Check RGB-related timestamps
    if not compare_timestamps(rgb_files, aolp_files, rgb_dir, aolp_dir):
        rgb_valid = False
    if not compare_timestamps(rgb_files, dolp_files, rgb_dir, dolp_dir):
        rgb_valid = False

    # Check segmentation-related timestamps
    if seg_files:
        if not compare_timestamps(rgb_files, seg_files, rgb_dir, seg_dir):
            seg_full = False

    # Check depth-related timestamps
    if depth_files and normal_files:
        if not compare_timestamps(depth_files, normal_files, depth_dir, normal_dir):
            depth_valid = False

    if not rgb_valid:
        print(f"\nError: RGB, AOLP, and DOLP 时间戳不一致")
        sys.exit(1)
    if not depth_valid and depth_files and normal_files:
        print(f"Error: Depth and Normal 时间戳不一致")
        sys.exit(1)
    if not seg_full:
        print(f"分割数据不完整，但不影响txt文件写入")

    # Write output files
    print("\nWriting output files...")
    write_timestamp_file(rgb_files, rgb_dir, os.path.join(base_directory, 'rgb.txt'))
    write_timestamp_file(aolp_files, aolp_dir, os.path.join(base_directory, 'aolp.txt'))
    write_timestamp_file(dolp_files, dolp_dir, os.path.join(base_directory, 'dolp.txt'))
    if normal_files:
        write_timestamp_file(normal_files, normal_dir, os.path.join(base_directory, 'normal.txt'))
    if depth_files:
        write_timestamp_file(depth_files, depth_dir, os.path.join(base_directory, 'depth.txt'))
    if seg_files:
        write_timestamp_file(seg_files, seg_dir, os.path.join(base_directory, 'seg.txt'))

    print("Processing complete!")


if __name__ == "__main__":
    main()