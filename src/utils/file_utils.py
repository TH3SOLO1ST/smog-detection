"""
File utilities for Islamabad Smog Detection System.

This module provides functions for file operations, naming conventions,
data validation, compression, and file management specific to the
satellite data processing workflow.
"""

import os
import hashlib
import gzip
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, BinaryIO
import datetime
import logging
import tarfile
import zipfile
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a processed file."""
    path: Path
    size_bytes: int
    checksum: str
    created_time: datetime.datetime
    modified_time: datetime.datetime
    file_type: str
    metadata: Dict[str, Any]


class FileUtils:
    """File utility functions for satellite data processing."""

    # File naming conventions
    NAMING_PATTERNS = {
        'raw_satellite': "{source}_{product}_{date}_{time}.{ext}",
        'processed': "islamabad_{product}_{date}_processed.{ext}",
        'timeseries': "islamabad_{product}_{freq}_{start_date}_{end_date}.{ext}",
        'export': "islamabad_smog_analysis_{date}_{time}.{ext}",
        'visualization': "islamabad_{viz_type}_{product}_{date}.{ext}"
    }

    # Supported file extensions and their types
    FILE_TYPES = {
        '.nc': 'netcdf',
        '.hdf': 'hdf4',
        '.h5': 'hdf5',
        '.tif': 'geotiff',
        '.tiff': 'geotiff',
        '.csv': 'csv',
        '.json': 'json',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.pdf': 'pdf',
        '.html': 'html',
        '.parquet': 'parquet',
        '.pkl': 'pickle',
        '.feather': 'feather'
    }

    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't.

        Args:
            directory: Directory path

        Returns:
            Path object for the directory
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @staticmethod
    def generate_filename(pattern_name: str, **kwargs) -> str:
        """
        Generate filename using specified naming pattern.

        Args:
            pattern_name: Name of pattern from NAMING_PATTERNS
            **kwargs: Variables to substitute in pattern

        Returns:
            Generated filename
        """
        if pattern_name not in FileUtils.NAMING_PATTERNS:
            raise ValueError(f"Unknown naming pattern: {pattern_name}")

        pattern = FileUtils.NAMING_PATTERNS[pattern_name]

        # Add current datetime if not provided
        if 'date' not in kwargs:
            kwargs['date'] = datetime.datetime.now().strftime('%Y%m%d')
        if 'time' not in kwargs:
            kwargs['time'] = datetime.datetime.now().strftime('%H%M%S')

        try:
            return pattern.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for filename pattern: {e}")

    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> FileInfo:
        """
        Get comprehensive information about a file.

        Args:
            file_path: Path to file

        Returns:
            FileInfo object with file details
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Basic file info
        stat = path.stat()
        size_bytes = stat.st_size
        created_time = datetime.datetime.fromtimestamp(stat.st_ctime)
        modified_time = datetime.datetime.fromtimestamp(stat.st_mtime)

        # File type
        file_type = FileUtils.FILE_TYPES.get(path.suffix.lower(), 'unknown')

        # Calculate checksum
        checksum = FileUtils.calculate_checksum(path)

        # Extract basic metadata
        metadata = {
            'name': path.name,
            'extension': path.suffix,
            'parent': str(path.parent),
            'is_absolute': path.is_absolute()
        }

        return FileInfo(
            path=path,
            size_bytes=size_bytes,
            checksum=checksum,
            created_time=created_time,
            modified_time=modified_time,
            file_type=file_type,
            metadata=metadata
        )

    @staticmethod
    def calculate_checksum(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
        """
        Calculate file checksum.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            Hexadecimal checksum string
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        hash_obj = hashlib.new(algorithm)

        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    @staticmethod
    def compress_file(input_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None,
                     compression_level: int = 6) -> Path:
        """
        Compress file using gzip.

        Args:
            input_path: Path to file to compress
            output_path: Output path (if None, adds .gz extension)
            compression_level: Compression level (1-9)

        Returns:
            Path to compressed file
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_path.with_suffix(input_path.suffix + '.gz')
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        FileUtils.ensure_directory(output_path.parent)

        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                shutil.copyfileobj(f_in, f_out)

        logger.info(f"Compressed {input_path} to {output_path}")
        return output_path

    @staticmethod
    def decompress_file(input_path: Union[str, Path],
                       output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Decompress gzip file.

        Args:
            input_path: Path to compressed file
            output_path: Output path (if None, removes .gz extension)

        Returns:
            Path to decompressed file
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            if input_path.suffix == '.gz':
                output_path = input_path.with_suffix('')
            else:
                raise ValueError("Cannot determine output path for decompression")
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        FileUtils.ensure_directory(output_path.parent)

        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        logger.info(f"Decompressed {input_path} to {output_path}")
        return output_path

    @staticmethod
    def save_dataframe(df: pd.DataFrame,
                      file_path: Union[str, Path],
                      format: str = 'parquet',
                      compression: Optional[str] = 'gzip',
                      **kwargs) -> Path:
        """
        Save pandas DataFrame to file.

        Args:
            df: DataFrame to save
            file_path: Output file path
            format: File format ('parquet', 'csv', 'json', 'feather', 'excel')
            compression: Compression method
            **kwargs: Additional arguments for saving function

        Returns:
            Path to saved file
        """
        file_path = Path(file_path)
        FileUtils.ensure_directory(file_path.parent)

        if format == 'parquet':
            df.to_parquet(file_path, compression=compression, **kwargs)
        elif format == 'csv':
            df.to_csv(file_path, compression=compression, **kwargs)
        elif format == 'json':
            df.to_json(file_path, compression=compression, **kwargs)
        elif format == 'feather':
            df.to_feather(file_path, **kwargs)
        elif format == 'excel':
            df.to_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved DataFrame to {file_path}")
        return file_path

    @staticmethod
    def load_dataframe(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load pandas DataFrame from file.

        Args:
            file_path: Path to file
            **kwargs: Additional arguments for loading function

        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = FileUtils.FILE_TYPES.get(file_path.suffix.lower(), 'unknown')

        if file_type == 'parquet':
            return pd.read_parquet(file_path, **kwargs)
        elif file_type == 'csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_type == 'json':
            return pd.read_json(file_path, **kwargs)
        elif file_type == 'feather':
            return pd.read_feather(file_path, **kwargs)
        elif file_type == 'pickle':
            return pd.read_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def save_numpy_array(array: np.ndarray,
                        file_path: Union[str, Path],
                        format: str = 'npy') -> Path:
        """
        Save numpy array to file.

        Args:
            array: Numpy array to save
            file_path: Output file path
            format: File format ('npy', 'npz', 'txt')

        Returns:
            Path to saved file
        """
        file_path = Path(file_path)
        FileUtils.ensure_directory(file_path.parent)

        if format == 'npy':
            np.save(file_path, array)
        elif format == 'npz':
            np.savez_compressed(file_path, array=array)
        elif format == 'txt':
            np.savetxt(file_path, array)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved numpy array to {file_path}")
        return file_path

    @staticmethod
    def load_numpy_array(file_path: Union[str, Path]) -> np.ndarray:
        """
        Load numpy array from file.

        Args:
            file_path: Path to file

        Returns:
            Loaded numpy array
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix == '.npz':
            data = np.load(file_path)
            return data['array']
        elif file_path.suffix == '.npy':
            return np.load(file_path)
        elif file_path.suffix == '.txt':
            return np.loadtxt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    @staticmethod
    def create_archive(input_paths: List[Union[str, Path]],
                      output_path: Union[str, Path],
                      format: str = 'tar.gz') -> Path:
        """
        Create archive from multiple files.

        Args:
            input_paths: List of input files/directories
            output_path: Output archive path
            format: Archive format ('tar.gz', 'tar', 'zip')

        Returns:
            Path to created archive
        """
        output_path = Path(output_path)
        FileUtils.ensure_directory(output_path.parent)

        if format == 'tar.gz' or format == 'tar':
            mode = 'w:gz' if format == 'tar.gz' else 'w'
            with tarfile.open(output_path, mode) as tar:
                for path in input_paths:
                    path = Path(path)
                    if path.is_file():
                        tar.add(path, arcname=path.name)
                    elif path.is_dir():
                        tar.add(path, arcname=path.name)

        elif format == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for path in input_paths:
                    path = Path(path)
                    if path.is_file():
                        zipf.write(path, arcname=path.name)
                    elif path.is_dir():
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                file_path = Path(root) / file
                                arcname = file_path.relative_to(path.parent)
                                zipf.write(file_path, arcname=arcname)
        else:
            raise ValueError(f"Unsupported archive format: {format}")

        logger.info(f"Created archive {output_path} from {len(input_paths)} inputs")
        return output_path

    @staticmethod
    def extract_archive(archive_path: Union[str, Path],
                       extract_to: Union[str, Path]) -> Path:
        """
        Extract archive to directory.

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to

        Returns:
            Path to extraction directory
        """
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)
        FileUtils.ensure_directory(extract_to)

        if archive_path.suffix in ['.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif archive_path.suffix == '.tar':
            with tarfile.open(archive_path, 'r:') as tar:
                tar.extractall(extract_to)
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

        logger.info(f"Extracted {archive_path} to {extract_to}")
        return extract_to

    @staticmethod
    def find_files(directory: Union[str, Path],
                  pattern: str = '*',
                  recursive: bool = True) -> List[Path]:
        """
        Find files matching pattern in directory.

        Args:
            directory: Directory to search
            pattern: Glob pattern to match
            recursive: Search recursively

        Returns:
            List of matching file paths
        """
        directory = Path(directory)

        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))

    @staticmethod
    def cleanup_old_files(directory: Union[str, Path],
                         older_than_days: int = 30,
                         pattern: str = '*') -> List[Path]:
        """
        Clean up old files in directory.

        Args:
            directory: Directory to clean
            older_than_days: Remove files older than this many days
            pattern: Glob pattern to match

        Returns:
            List of deleted file paths
        """
        directory = Path(directory)
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        deleted_files = []

        for file_path in FileUtils.find_files(directory, pattern):
            if file_path.is_file():
                modified_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                if modified_time < cutoff_date:
                    try:
                        file_path.unlink()
                        deleted_files.append(file_path)
                        logger.info(f"Deleted old file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")

        return deleted_files

    @staticmethod
    def validate_file_integrity(file_path: Union[str, Path],
                               expected_checksum: Optional[str] = None) -> bool:
        """
        Validate file integrity.

        Args:
            file_path: Path to file
            expected_checksum: Expected MD5 checksum (optional)

        Returns:
            True if file is valid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False

        # Check if file is empty
        if file_path.stat().st_size == 0:
            logger.error(f"File is empty: {file_path}")
            return False

        # Verify checksum if provided
        if expected_checksum is not None:
            actual_checksum = FileUtils.calculate_checksum(file_path)
            if actual_checksum != expected_checksum:
                logger.error(f"Checksum mismatch for {file_path}: "
                           f"expected {expected_checksum}, got {actual_checksum}")
                return False

        return True

    @staticmethod
    def get_directory_size(directory: Union[str, Path]) -> Dict[str, int]:
        """
        Get directory size information.

        Args:
            directory: Directory path

        Returns:
            Dictionary with size information
        """
        directory = Path(directory)

        if not directory.exists():
            return {'total_bytes': 0, 'file_count': 0, 'directory_count': 0}

        total_bytes = 0
        file_count = 0
        directory_count = 0

        for item in directory.rglob('*'):
            if item.is_file():
                total_bytes += item.stat().st_size
                file_count += 1
            elif item.is_dir():
                directory_count += 1

        return {
            'total_bytes': total_bytes,
            'file_count': file_count,
            'directory_count': directory_count,
            'total_mb': round(total_bytes / (1024 * 1024), 2),
            'total_gb': round(total_bytes / (1024 * 1024 * 1024), 3)
        }