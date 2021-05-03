from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataProfilingConfig:
    input_data_path: str = MISSING
    report_dir: str = MISSING
    report_file_name: str = MISSING
    title: str = MISSING
