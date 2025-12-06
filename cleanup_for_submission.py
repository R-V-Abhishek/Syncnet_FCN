#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cleanup_for_submission.py - Prepare repository for submission

This script cleans up unnecessary files while preserving the best trained model.

Usage:
    # Dry run (shows what would be deleted)
    python cleanup_for_submission.py --dry-run
    
    # Actually clean up
    python cleanup_for_submission.py --execute
    
    # Keep only the best model checkpoint
    python cleanup_for_submission.py --execute --keep-best

Author: R V Abhishek
"""

import os
import shutil
import argparse
import glob

# Directories to clean
CLEANUP_DIRS = [
    'temp_dataset',
    'temp',
    'temp_eval',
    'temp_hls',
    '__pycache__',
    '.history',
    'data/work',
]

# File patterns to remove
CLEANUP_PATTERNS = [
    '*.pyc',
    '*.pyo',
    '*.tmp',
    '*.temp',
    '*_audio.wav',  # Temp audio files
]

# Checkpoint directories
CHECKPOINT_DIRS = [
    'checkpoints',
    'checkpoints_attention', 
    'checkpoints_regression',
]

# Files to keep (important)
KEEP_FILES = [
    'syncnet_fcn_best.pth',  # Best trained model
    'syncnet_v2.model',       # Pretrained base model
]


def get_size_mb(path):
    """Get size of file or directory in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return total / (1024 * 1024)
    return 0


def cleanup(dry_run=True, keep_best=True, verbose=True):
    """
    Clean up unnecessary files.
    
    Args:
        dry_run: If True, only show what would be deleted
        keep_best: If True, keep the best checkpoint
        verbose: Print detailed info
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*60)
    print("FCN-SyncNet Cleanup Script")
    print("="*60)
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"Keep best model: {keep_best}")
    print()
    
    total_size = 0
    items_to_remove = []
    
    # 1. Clean temp directories
    print("📁 Temporary Directories:")
    for dir_name in CLEANUP_DIRS:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            size = get_size_mb(dir_path)
            total_size += size
            items_to_remove.append(('dir', dir_path))
            print(f"   [DELETE] {dir_name}/ ({size:.2f} MB)")
        else:
            if verbose:
                print(f"   [SKIP] {dir_name}/ (not found)")
    print()
    
    # 2. Clean file patterns
    print("📄 Temporary Files:")
    for pattern in CLEANUP_PATTERNS:
        matches = glob.glob(os.path.join(base_dir, '**', pattern), recursive=True)
        for match in matches:
            size = get_size_mb(match)
            total_size += size
            items_to_remove.append(('file', match))
            rel_path = os.path.relpath(match, base_dir)
            print(f"   [DELETE] {rel_path} ({size:.2f} MB)")
    print()
    
    # 3. Handle checkpoints
    print("🔧 Checkpoint Directories:")
    for ckpt_dir in CHECKPOINT_DIRS:
        ckpt_path = os.path.join(base_dir, ckpt_dir)
        if os.path.exists(ckpt_path):
            # List checkpoint files
            ckpt_files = glob.glob(os.path.join(ckpt_path, '*.pth'))
            
            for ckpt_file in ckpt_files:
                filename = os.path.basename(ckpt_file)
                size = get_size_mb(ckpt_file)
                
                # Keep best model if requested
                if keep_best and filename in KEEP_FILES:
                    print(f"   [KEEP] {ckpt_dir}/{filename} ({size:.2f} MB)")
                else:
                    total_size += size
                    items_to_remove.append(('file', ckpt_file))
                    print(f"   [DELETE] {ckpt_dir}/{filename} ({size:.2f} MB)")
    print()
    
    # Summary
    print("="*60)
    print(f"Total space to free: {total_size:.2f} MB")
    print(f"Items to remove: {len(items_to_remove)}")
    print("="*60)
    
    if dry_run:
        print("\n⚠️  DRY RUN - No files were deleted.")
        print("    Run with --execute to actually delete files.")
        return
    
    # Confirm
    if not dry_run:
        confirm = input("\n⚠️  Are you sure you want to delete these files? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            return
    
    # Execute cleanup
    print("\n🧹 Cleaning up...")
    deleted_count = 0
    error_count = 0
    
    for item_type, item_path in items_to_remove:
        try:
            if item_type == 'dir':
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
            deleted_count += 1
            if verbose:
                print(f"   ✓ Deleted: {os.path.relpath(item_path, base_dir)}")
        except Exception as e:
            error_count += 1
            print(f"   ✗ Error deleting {item_path}: {e}")
    
    print()
    print("="*60)
    print(f"✅ Cleanup complete!")
    print(f"   Deleted: {deleted_count} items")
    print(f"   Errors: {error_count}")
    print(f"   Space freed: ~{total_size:.2f} MB")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Cleanup script for submission')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Show what would be deleted without deleting (default)')
    parser.add_argument('--execute', action='store_true',
                       help='Actually delete files')
    parser.add_argument('--keep-best', action='store_true', default=True,
                       help='Keep the best model checkpoint (default: True)')
    parser.add_argument('--delete-all-checkpoints', action='store_true',
                       help='Delete ALL checkpoints including best model')
    parser.add_argument('--quiet', action='store_true',
                       help='Less verbose output')
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    keep_best = not args.delete_all_checkpoints
    verbose = not args.quiet
    
    cleanup(dry_run=dry_run, keep_best=keep_best, verbose=verbose)


if __name__ == '__main__':
    main()
