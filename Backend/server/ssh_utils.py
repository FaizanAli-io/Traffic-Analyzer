import socket
import paramiko
from scp import SCPClient
from .config import SSH_KEY_PATH, SSH_USERNAME


def load_ssh_key():
    try:
        return paramiko.RSAKey.from_private_key_file(SSH_KEY_PATH)
    except Exception:
        return paramiko.Ed25519Key.from_private_key_file(SSH_KEY_PATH)


def make_ssh(ip):
    key = load_ssh_key()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username=SSH_USERNAME, pkey=key, timeout=30)
    scp = SCPClient(ssh.get_transport())
    return ssh, scp


def sh(ssh, cmd):
    cmd = f"bash -lc '{cmd}'"
    _, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode("utf-8", "ignore")
    err = stderr.read().decode("utf-8", "ignore")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def wait_for_ssh(ip, port=22, timeout=300):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((ip, port), timeout=5):
                return True
        except OSError:
            time.sleep(5)
    return False
