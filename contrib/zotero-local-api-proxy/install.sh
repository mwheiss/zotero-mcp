#!/usr/bin/env bash
set -euo pipefail

UNIT_NAME="zotero-local-api-proxy"
LISTEN_ADDRESS="0.0.0.0"
LISTEN_PORT="23120"
TARGET_ADDRESS="127.0.0.1"
TARGET_PORT="23119"
INSTALL_ROOT="/"
UNINSTALL=false

usage() {
    cat <<'EOF'
Install a boot-persistent LAN listener for Zotero's loopback Local API.

Usage:
  sudo ./install.sh [options]
  sudo ./install.sh --uninstall

Options:
  --listen-address ADDRESS  Address exposed to the LAN (default: 0.0.0.0)
  --listen-port PORT        LAN port (default: 23120)
  --target-address ADDRESS  Zotero loopback address (default: 127.0.0.1)
  --target-port PORT        Zotero Local API port (default: 23119)
  --root PATH               Alternate install root for packaging/testing
  --uninstall               Disable and remove the installed units
  -h, --help                Show this help

Example:
  sudo ./install.sh --listen-address 192.168.1.50 --listen-port 23120
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

validate_address() {
    local value="$1"
    [[ -n "$value" ]] || die "address must not be empty"
    [[ "$value" =~ ^[A-Za-z0-9._:-]+$ ]] || die "invalid address: $value"
}

validate_port() {
    local value="$1"
    [[ "$value" =~ ^[0-9]+$ ]] || die "port must be numeric: $value"
    (( value >= 1 && value <= 65535 )) || die "port out of range: $value"
}

format_stream() {
    local address="$1"
    local port="$2"
    if [[ "$address" == *:* && "$address" != \[*\] ]]; then
        printf '[%s]:%s' "$address" "$port"
    else
        printf '%s:%s' "$address" "$port"
    fi
}

while (( $# )); do
    case "$1" in
        --listen-address)
            (( $# >= 2 )) || die "--listen-address requires a value"
            LISTEN_ADDRESS="$2"
            shift 2
            ;;
        --listen-port)
            (( $# >= 2 )) || die "--listen-port requires a value"
            LISTEN_PORT="$2"
            shift 2
            ;;
        --target-address)
            (( $# >= 2 )) || die "--target-address requires a value"
            TARGET_ADDRESS="$2"
            shift 2
            ;;
        --target-port)
            (( $# >= 2 )) || die "--target-port requires a value"
            TARGET_PORT="$2"
            shift 2
            ;;
        --root)
            (( $# >= 2 )) || die "--root requires a value"
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

validate_address "$LISTEN_ADDRESS"
validate_address "$TARGET_ADDRESS"
validate_port "$LISTEN_PORT"
validate_port "$TARGET_PORT"

if [[ "$INSTALL_ROOT" == "/" && ${EUID:-$(id -u)} -ne 0 ]]; then
    die "run this installer as root (for example: sudo ./install.sh)"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR="${INSTALL_ROOT%/}/etc/systemd/system"
SOCKET_PATH="$SYSTEMD_DIR/$UNIT_NAME.socket"
SERVICE_PATH="$SYSTEMD_DIR/$UNIT_NAME.service"

manage_systemd=true
if [[ "$INSTALL_ROOT" != "/" ]]; then
    manage_systemd=false
fi

if [[ "$UNINSTALL" == true ]]; then
    if [[ "$manage_systemd" == true ]]; then
        systemctl disable --now "$UNIT_NAME.socket" 2>/dev/null || true
        systemctl stop "$UNIT_NAME.service" 2>/dev/null || true
    fi
    rm -f -- "$SOCKET_PATH" "$SERVICE_PATH"
    if [[ "$manage_systemd" == true ]]; then
        systemctl daemon-reload
        systemctl reset-failed "$UNIT_NAME.service" "$UNIT_NAME.socket" 2>/dev/null || true
    fi
    printf 'Removed %s.\n' "$UNIT_NAME"
    exit 0
fi

PROXY_BINARY=""
for candidate in \
    /usr/lib/systemd/systemd-socket-proxyd \
    /lib/systemd/systemd-socket-proxyd \
    "$(command -v systemd-socket-proxyd 2>/dev/null || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        PROXY_BINARY="$candidate"
        break
    fi
done
[[ -n "$PROXY_BINARY" ]] || die "systemd-socket-proxyd was not found; install/update the systemd package"

LISTEN_STREAM="$(format_stream "$LISTEN_ADDRESS" "$LISTEN_PORT")"
TARGET_STREAM="$(format_stream "$TARGET_ADDRESS" "$TARGET_PORT")"

mkdir -p -- "$SYSTEMD_DIR"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT

sed \
    -e "s|@LISTEN_STREAM@|$LISTEN_STREAM|g" \
    "$SCRIPT_DIR/$UNIT_NAME.socket.in" > "$TEMP_DIR/$UNIT_NAME.socket"
sed \
    -e "s|@PROXY_BINARY@|$PROXY_BINARY|g" \
    -e "s|@TARGET_STREAM@|$TARGET_STREAM|g" \
    "$SCRIPT_DIR/$UNIT_NAME.service.in" > "$TEMP_DIR/$UNIT_NAME.service"

install -m 0644 "$TEMP_DIR/$UNIT_NAME.socket" "$SOCKET_PATH"
install -m 0644 "$TEMP_DIR/$UNIT_NAME.service" "$SERVICE_PATH"

if [[ "$manage_systemd" == true ]]; then
    systemctl disable --now "$UNIT_NAME.socket" 2>/dev/null || true
    systemctl stop "$UNIT_NAME.service" 2>/dev/null || true
    systemctl daemon-reload
    systemctl enable --now "$UNIT_NAME.socket"
fi

cat <<EOF
Installed $UNIT_NAME.

LAN listener:  $LISTEN_STREAM
Zotero target: $TARGET_STREAM
MCP setting:   ZOTERO_REMOTE_LOCAL_URL=http://<CACHYOS-LAN-IP>:$LISTEN_PORT/api

Zotero must be running with local API access enabled. If a firewall is active,
allow TCP port $LISTEN_PORT from the MCP server's LAN address.

Verify from the MCP server:
  curl -i -H 'Host: 127.0.0.1:$TARGET_PORT' \\
    http://<CACHYOS-LAN-IP>:$LISTEN_PORT/api/

Then authorize the selected endpoint:
  ZOTERO_REMOTE_LOCAL_URL=http://<CACHYOS-LAN-IP>:$LISTEN_PORT/api \\
    zotero-mcp authorize-local-writes
EOF
