import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import threading
from typing import Any, Dict, Optional


class DeviceProcessManager:
    def __init__(self, project_dir: Optional[str] = None, verbose: bool = False):
        self.verbose = verbose
        self.project_dir = project_dir or os.getcwd()
        self.process: Optional[subprocess.Popen] = None
        self.stdout_thread: Optional[threading.Thread] = None
        self.stdout_queue: queue.Queue = queue.Queue()
        self.ready_event = threading.Event()
        self.last_lines = []
        self.lock = threading.Lock()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _stream_output(self) -> None:
        assert self.process is not None
        seen_listening = False
        seen_power = False

        while True:
            line = self.process.stdout.readline()
            if not line:
                break

            stripped = line.rstrip("\n")
            self.stdout_queue.put(stripped)
            self.last_lines.append(stripped)
            if len(self.last_lines) > 200:
                self.last_lines.pop(0)

            # Mirror subprocess output to manager stdout for easier debugging.
            print(f"[device.py] {stripped}", flush=True)

            if "is listening on host:port" in stripped:
                seen_listening = True
            if "Recording power and temperature" in stripped:
                seen_power = True

            if seen_listening and seen_power:
                self.ready_event.set()

    def start(self, payload: Dict[str, Any], timeout_s: int = 60) -> Dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return {
                    "ok": False,
                    "error": "Device process already running.",
                    "pid": self.process.pid,
                }

            required = ["host", "port", "device_type", "dev_idx", "exp", "run", "seed"]
            missing = [k for k in required if k not in payload]
            if missing:
                return {"ok": False, "error": f"Missing payload keys: {missing}"}

            cmd = [
                sys.executable,
                "-u",
                "device.py",
                "--host",
                str(payload["host"]),
                "--port",
                str(payload["port"]),
                "--device_type",
                str(payload["device_type"]),
                "--dev_idx",
                str(payload["dev_idx"]),
                "--exp",
                str(payload["exp"]),
                "--r",
                str(payload["run"]),
                "--seed",
                str(payload["seed"]),
            ]
            if bool(payload.get("verbose", False)):
                cmd.append("--verbose")

            cwd = payload.get("project_dir") or self.project_dir
            self.ready_event.clear()
            self.last_lines = []

            self._log(f"[+] Starting device process: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            self.stdout_thread = threading.Thread(target=self._stream_output, daemon=True)
            self.stdout_thread.start()

        ready = self.ready_event.wait(timeout=timeout_s)

        with self.lock:
            if self.process is None:
                return {"ok": False, "error": "Process missing after launch."}

            if ready:
                return {"ok": True, "message": "Device ready.", "pid": self.process.pid}

            returncode = self.process.poll()
            if returncode is not None:
                return {
                    "ok": False,
                    "error": f"Device process exited early with code {returncode}.",
                    "tail": self.last_lines[-20:],
                }

            return {
                "ok": False,
                "error": f"Timed out waiting for readiness ({timeout_s}s).",
                "tail": self.last_lines[-20:],
            }

    def status(self) -> Dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {
                "ok": True,
                "running": running,
                "ready": self.ready_event.is_set() if running else False,
                "pid": self.process.pid if running else None,
            }

    def shutdown_process(self, grace_s: int = 5) -> Dict[str, Any]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                return {"ok": True, "message": "No running process."}

            self._log("[+] Terminating running device process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                self._log("[!] Device process did not terminate; killing...")
                self.process.kill()
                self.process.wait(timeout=grace_s)

            return {"ok": True, "message": "Device process terminated."}


def _read_json_line(connection: socket.socket) -> Optional[Dict[str, Any]]:
    data = b""
    while b"\n" not in data:
        chunk = connection.recv(4096)
        if not chunk:
            break
        data += chunk

    if not data:
        return None

    try:
        payload = json.loads(data.decode("utf-8").strip())
        if not isinstance(payload, dict):
            return None
        return payload
    except json.JSONDecodeError:
        return None


def _send_json_line(connection: socket.socket, payload: Dict[str, Any]) -> None:
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    connection.sendall(raw)


def serve(listen_host: str, listen_port: int, project_dir: Optional[str] = None, verbose: bool = False) -> None:
    manager = DeviceProcessManager(project_dir=project_dir, verbose=verbose)
    server_stop_event = threading.Event()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((listen_host, listen_port))
        server.listen(8)
        print(f"[+] Device Manager listening on {listen_host}:{listen_port}", flush=True)

        while not server_stop_event.is_set():
            conn, addr = server.accept()
            with conn:
                req = _read_json_line(conn)
                if req is None:
                    _send_json_line(conn, {"ok": False, "error": "Invalid request."})
                    continue

                cmd = req.get("cmd", "").upper()
                if cmd == "PING":
                    _send_json_line(conn, {"ok": True, "message": "pong"})
                elif cmd == "PREPARE":
                    payload = req.get("payload", {})
                    timeout_s = int(req.get("timeout_s", 90))
                    res = manager.start(payload=payload, timeout_s=timeout_s)
                    _send_json_line(conn, res)
                elif cmd == "STATUS":
                    _send_json_line(conn, manager.status())
                elif cmd == "SHUTDOWN":
                    _send_json_line(conn, manager.shutdown_process())
                elif cmd == "STOP_SERVER":
                    shutdown_result = manager.shutdown_process()
                    _send_json_line(conn, {"ok": True, "message": "Server stopping.", "shutdown": shutdown_result})
                    server_stop_event.set()
                else:
                    _send_json_line(conn, {"ok": False, "error": f"Unknown cmd: {cmd}"})

            if verbose:
                print(f"[+] Handled request from {addr[0]}:{addr[1]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Device Manager service for queued FL jobs")
    parser.add_argument("--listen_host", type=str, default="0.0.0.0", help="Host/IP for device manager service")
    parser.add_argument("--listen_port", type=int, default=10090, help="Port for device manager service")
    parser.add_argument("--project_dir", type=str, default=None, help="Path containing device.py on this device")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    serve(listen_host=args.listen_host, listen_port=args.listen_port, project_dir=args.project_dir, verbose=args.verbose)
