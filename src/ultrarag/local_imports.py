"""Import server-owned Python modules into a private session namespace."""

from __future__ import annotations

import builtins
import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class ServerModuleNamespace(importlib.abc.MetaPathFinder):
    """Keep ordinary sibling imports independent for each server instance.

    Third-party modules keep Python's normal shared import cache. Modules under
    the server's source directory, including their relative imports, belong to
    this namespace and are discarded when the execution client closes.
    """

    def __init__(self, name: str, directory: Path) -> None:
        self.name = name
        self.directory = directory
        self.importlib_proxy = ModuleType("importlib")
        self.importlib_proxy.__dict__["__getattr__"] = lambda key: getattr(
            importlib, key
        )
        self.importlib_proxy.import_module = self.import_module
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        sys.modules[name] = package
        sys.meta_path.insert(0, self)

    def _import(
        self,
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "importlib" or (name.startswith("importlib.") and not fromlist):
            builtins.__import__(name, globals, locals, fromlist, level)
            return self.importlib_proxy
        root = name.split(".", 1)[0]
        if level == 0 and (
            (self.directory / f"{root}.py").is_file()
            or (self.directory / root).is_dir()
        ):
            qualified = f"{self.name}.{name}"
            result = builtins.__import__(qualified, globals, locals, fromlist, 0)
            return result if fromlist else sys.modules[f"{self.name}.{root}"]
        return builtins.__import__(name, globals, locals, fromlist, level)

    def import_module(self, name: str, package: str | None = None) -> Any:
        root = name.split(".", 1)[0]
        if not name.startswith(".") and (
            (self.directory / f"{root}.py").is_file()
            or (self.directory / root).is_dir()
        ):
            name = f"{self.name}.{name}"
        return importlib.import_module(name, package)

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        prefix = f"{self.name}."
        if not fullname.startswith(prefix):
            return None
        relative = fullname[len(prefix) :].split(".")
        source = self.directory.joinpath(*relative)
        is_package = source.is_dir()
        source = source / "__init__.py" if is_package else source.with_suffix(".py")
        if not source.is_file():
            return None
        loader = _ServerModuleLoader(self, source)
        return importlib.util.spec_from_file_location(
            fullname,
            source,
            loader=loader,
            submodule_search_locations=[str(source.parent)] if is_package else None,
        )

    def prepare(self, module: ModuleType) -> None:
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=self._import)

    def close(self) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        for name in list(sys.modules):
            if name == self.name or name.startswith(f"{self.name}."):
                sys.modules.pop(name, None)


class _ServerModuleLoader(importlib.abc.Loader):
    def __init__(self, namespace: ServerModuleNamespace, source: Path) -> None:
        self.namespace = namespace
        self.source = source

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.namespace.prepare(module)
        # compile(bytes) honors Python source encoding cookies.
        exec(  # noqa: S102 - load the user-selected module
            compile(self.source.read_bytes(), str(self.source), "exec"), module.__dict__
        )
