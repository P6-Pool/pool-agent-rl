"""
Model Utilities
"""

import os
import re
import json
from typing import Union, Optional
from dataclasses import dataclass


@dataclass(init=False)
class ModelMetadata:
    name: Optional[str]
    version: Optional[int]
    model_dir: Optional[str]
    logs_dir: Optional[str]

    def __init__(
        self,
        name: Optional[str] = None,
        version: Optional[int] = None,
        model_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.model_dir = model_dir
        self.logs_dir = logs_dir


class ModelManager:
    """
    Used to manage models
    """

    def __init__(
        self,
        name: Optional[str],
        model_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
    ) -> None:
        self.name_handler = ModelNameHandler.from_str(name)

        # if not os.path.exists(model_dir):
        #     raise FileNotFoundError(f"Model directory '{model_dir}' does not exist")

        # if not os.path.exists(logs_dir):
        #     raise FileNotFoundError(f"Logs directory '{logs_dir}' does not exist")

        self.model_dir = model_dir
        self.logs_dir = logs_dir
        self.metadata = ModelMetadata(
            self.name_handler.name, self.name_handler.version, model_dir, logs_dir
        )

    def __str__(self) -> str:
        return str(self.name_handler)

    def latest_version(self) -> Optional[int]:
        """
        Get the version of latest version of the model
        """
        if not os.path.exists(self.model_dir):
            return None

        pattern = re.compile(r".*-v(?P<VERSION>[0-9]+).*")
        for file in os.listdir(self.model_dir):
            if file.startswith(self.name_handler.base_name):
                try:
                    version = int(pattern.match(file).group("VERSION"))
                except Exception:
                    continue
                else:
                    if self.name_handler.version is None:
                        self.name_handler.version = version
                    elif version > self.name_handler.version:
                        self.name_handler.version = version

        return self.name_handler.version

    def next_version(self) -> int:
        """
        Get the next version of the model
        """
        latest_version = self.latest_version()
        if latest_version is None:
            return 1
        return latest_version + 1

    @property
    def model_path(self) -> str:
        """
        Full path of the model
        """
        return os.path.join(self.model_dir, self.name_handler.name)

    @property
    def metadata_path(self) -> str:
        """
        Full path of the metadata file
        """
        return os.path.join(self.model_dir, f"{self.name_handler.name}-metadata.json")

    def save_metadata(self, metadata: Optional[ModelMetadata] = None) -> None:
        """
        Save the metadata
        """
        if metadata is None:
            metadata = self.metadata
            metadata.name = self.name_handler.name
            metadata.version = self.name_handler.version
            metadata.model_dir = self.model_dir
            metadata.logs_dir = self.logs_dir

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        with open(self.metadata_path, "w+") as fp:
            json.dump(metadata.__dict__, fp, indent=4)


class ModelNameHandler:
    """
    Used to create model names
    """

    def __init__(
        self,
        base_name: str = "",
        postfix: Optional[str] = None,
        version: Optional[int] = None,
    ) -> None:
        self._base_name: str = base_name
        self._postfix: Optional[str] = postfix
        self._version: Optional[int] = version

        self._name: str = ""
        self._update_name()

    @classmethod
    def from_str(cls, name: str) -> "ModelNameHandler":
        name_list = name.split("-")
        version = None
        postfix = None

        if "v" in name_list[-1]:
            version = int(name_list[-1][1:])
            name_list.pop(-1)

        if len(name_list) > 1:
            postfix = name_list[-1]
            name_list.pop(-1)

        base_name = "-".join(name_list)
        return cls(base_name, postfix, version)

    def __str__(self) -> str:
        self._update_name()
        return self._name

    def _update_name(self) -> None:
        name_list = []

        name_list.append(self._base_name)

        if self._postfix is not None:
            name_list.append(self._postfix)

        if self._version is not None:
            name_list.append(f"v{self._version}")

        self._name = "-".join(name_list)

    @property
    def name(self) -> str:
        self._update_name()
        return self._name

    @property
    def base_name(self) -> str:
        return self._base_name

    @base_name.setter
    def base_name(self, base_name: str) -> None:
        self._base_name = base_name
        self._update_name()

    @property
    def postfix(self) -> str:
        return self._postfix

    @postfix.setter
    def postfix(self, postfix: Union[str, None]) -> None:
        self._postfix = postfix
        self._update_name()

    @property
    def version(self) -> int:
        return self._version

    @version.setter
    def version(self, version: Union[int, None]) -> None:
        self._version = version
        self._update_name()


if __name__ == "__main__":
    LOGS_DIR = "logs/"
    MODEL_DIR = "models/"

    mm = ModelManager("ppo-v1", MODEL_DIR, LOGS_DIR)
    mm.metadata = ModelMetadata("ppo", 1, MODEL_DIR, LOGS_DIR)
    print(mm.model_path)
    print(mm.metadata_path)
    mm.save_metadata()
