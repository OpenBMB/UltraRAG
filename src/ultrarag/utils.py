import ctypes
import logging
import os
import signal
import subprocess
from typing import Dict, List, Optional

IS_POSIX = os.name == "posix"
IS_WINDOWS = os.name == "nt"

if IS_POSIX:
    libc = ctypes.CDLL(None)
else:
    libc = None

_windows_job_handle = None

if IS_WINDOWS:
    import ctypes.wintypes as wintypes

    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def _windows_ensure_job_object():
        """Ensure Windows job object exists for process group management.

        Creates a job object that will kill child processes when the parent
        process terminates, ensuring proper cleanup.

        Returns:
            Job object handle

        Raises:
            OSError: If job object creation or configuration fails
        """
        global _windows_job_handle
        if _windows_job_handle:
            return _windows_job_handle

        hJob = kernel32.CreateJobObjectW(None, None)
        if not hJob:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        res = kernel32.SetInformationJobObject(
            hJob,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not res:
            err = ctypes.get_last_error()
            kernel32.CloseHandle(hJob)
            raise OSError(err, "SetInformationJobObject failed")

        _windows_job_handle = hJob
        return _windows_job_handle

    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.INT,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL


def set_pdeathsig() -> None:
    """Set parent death signal on POSIX systems.

    Configures the process to receive SIGTERM when the parent process dies,
    ensuring child processes are properly cleaned up.
    """
    if IS_POSIX and libc is not None:
        libc.prctl(1, signal.SIGTERM)


def popen_follow_parent(
    command: List[str], env: Optional[Dict[str, str]] = None
) -> subprocess.Popen:
    """Create a subprocess that will be terminated when parent dies.

    On POSIX systems, uses prctl to set parent death signal.
    On Windows, uses job objects to ensure child processes are killed
    when the parent terminates.

    Args:
        command: Command to execute as a list of strings
        env: Optional environment variables dictionary

    Returns:
        Popen object representing the started subprocess
    """
    if IS_POSIX:
        return subprocess.Popen(command, env=env, preexec_fn=set_pdeathsig)
    elif IS_WINDOWS:
        hJob = _windows_ensure_job_object()
        proc = subprocess.Popen(command, env=env)

        hProcess = wintypes.HANDLE(proc._handle)
        ok = kernel32.AssignProcessToJobObject(hJob, hProcess)
        if not ok:
            logging.warning(
                "AssignProcessToJobObject failed; child may outlive parent."
            )
        return proc
    else:
        return subprocess.Popen(command, env=env)
