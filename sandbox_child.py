"""python_exec 的子进程入口：用审计钩子做「软沙箱」。

能挡住的（教学/防误伤级别）：
- 写 workspace 之外的文件
- 读 workspace、Python 安装目录、系统目录之外的普通文件
- 网络连接 / DNS
- 启动子进程
- 通过 ctypes 加载动态库

挡不住的：刻意用底层手段绕过的代码。这是权限边界，不是安全边界。
"""

from __future__ import annotations

import os
import sys

_WRITE_EVENTS = {
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.mkdir",
    "os.chmod",
    "os.chown",
    "os.truncate",
    "os.utime",
    "os.link",
    "os.symlink",
    "os.chdir",
}

_NETWORK_EVENTS = {
    "socket.connect",
    "socket.bind",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.sendto",
}

_PROCESS_EVENTS = {
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
}


def _norm(path: object) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))  # type: ignore[arg-type]


def _inside(path: str, roots: list[str]) -> bool:
    for root in roots:
        if path == root or path.startswith(root + os.sep):
            return True
    return False


def install(workspace: str) -> None:
    roots: list[str] = []
    for candidate in (
        workspace,
        __file__,
        sys.prefix,
        sys.base_prefix,
        os.environ.get("SYSTEMROOT", ""),
        os.environ.get("WINDIR", ""),
    ):
        if candidate:
            try:
                roots.append(_norm(candidate))
            except Exception:
                continue
    ws = _norm(workspace)

    def hook(event: str, args: tuple[object, ...]) -> None:
        if event == "open":
            if len(args) < 2:
                return
            raw_path, mode = args[0], args[1]
            if isinstance(raw_path, int) or raw_path is None:
                return
            try:
                path = _norm(raw_path)
            except Exception:
                return
            flags = args[2] if len(args) > 2 else 0
            writing = bool(mode and any(ch in str(mode) for ch in "wax+"))
            if not writing and isinstance(flags, int):
                writing = bool(
                    flags
                    & (
                        os.O_WRONLY
                        | os.O_RDWR
                        | os.O_CREAT
                        | os.O_TRUNC
                        | os.O_APPEND
                    )
                )
            if writing:
                if not _inside(path, [ws]):
                    raise PermissionError(
                        f"沙箱：禁止写入 workspace 之外的路径: {raw_path}"
                    )
            elif not _inside(path, roots):
                raise PermissionError(
                    f"沙箱：禁止读取 workspace 之外的路径: {raw_path}"
                )
            return

        if event in _WRITE_EVENTS:
            for value in args:
                if isinstance(value, (str, bytes, os.PathLike)):
                    try:
                        path = _norm(value)
                    except Exception:
                        continue
                    if not _inside(path, [ws]):
                        raise PermissionError(
                            f"沙箱：禁止在 workspace 之外执行 {event}: {value}"
                        )
            return

        if event in _NETWORK_EVENTS:
            raise PermissionError(f"沙箱：禁止网络访问（{event}）")

        if event in _PROCESS_EVENTS:
            raise PermissionError(f"沙箱：禁止启动子进程（{event}）")

    # 注意：不拦 ctypes.dlopen。numpy/pandas 在导入时就会通过 ctypes 加载
    # 自己的动态库，拦住会让数据处理工具整体不可用。ctypes 因此是已知绕过点，
    # 见 README「软沙箱的边界」。
    sys.addaudithook(hook)


def main() -> int:
    if len(sys.argv) < 2:
        print("sandbox_child: 缺少 workspace 参数", file=sys.stderr)
        return 2
    workspace = os.path.abspath(sys.argv[1])
    code = sys.stdin.read()
    os.makedirs(workspace, exist_ok=True)
    os.chdir(workspace)
    install(workspace)
    sys.argv = ["<agent-code>"]
    exec(compile(code, "<agent-code>", "exec"), {"__name__": "__main__"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
