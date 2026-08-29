# Zotero Local API LAN proxy for CachyOS

This installs a socket-activated systemd listener that forwards a LAN port to
Zotero's loopback-only Local API at `127.0.0.1:23119`. The socket is enabled at
boot. `systemd-socket-proxyd` starts on demand and exits after five idle minutes.

## Install on the Zotero workstation

Make sure Zotero's **Allow other applications on this computer to communicate
with Zotero** setting is enabled, then run:

```bash
cd contrib/zotero-local-api-proxy
sudo ./install.sh --listen-address 0.0.0.0 --listen-port 23120
```

Binding `0.0.0.0` survives DHCP address changes. To restrict the listener to a
specific interface address, pass that address instead.

Check it with:

```bash
systemctl status zotero-local-api-proxy.socket
systemctl status zotero-local-api-proxy.service
ss -ltnp | grep 23120
```

From the MCP server:

```bash
curl -i -H 'Host: 127.0.0.1:23119' \
  http://CACHYOS_LAN_IP:23120/api/
```

The response should include `Zotero-API-Version` and `Zotero-Server-ID` headers.
Zotero validates the HTTP `Host` header even though the TCP listener is proxied;
Zotero MCP applies this loopback Host header automatically for remote-local
requests.

## Firewall

A firewall rule alone cannot expose a process bound to loopback; this proxy
provides the LAN listener. If a firewall is active, allow the proxy port from
the MCP server only.

For firewalld:

```bash
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="MCP_SERVER_IP/32" port port="23120" protocol="tcp" accept'
sudo firewall-cmd --reload
```

For UFW:

```bash
sudo ufw allow from MCP_SERVER_IP to any port 23120 proto tcp
```

If no host firewall is active, no firewall rule is needed. LAN HTTP is supported
by Zotero MCP, but the local authorization key is sent unencrypted.

## Configure Zotero MCP

Set this in the MCP server process environment:

```bash
ZOTERO_REMOTE_LOCAL_URL=http://CACHYOS_LAN_IP:23120/api
```

Restart the MCP service, then request authorization:

```bash
ZOTERO_REMOTE_LOCAL_URL=http://CACHYOS_LAN_IP:23120/api \
  zotero-mcp authorize-local-writes
```

The dialog appears in Zotero on the workstation. Choose **Always Allow** to
avoid a prompt for every write.

If another trusted application already stores a remembered key for the same
`Zotero-Server-ID`, that exact key can be shared explicitly instead:

```bash
ZOTERO_REMOTE_LOCAL_API_KEY=THE_EXISTING_32_CHARACTER_KEY
```

Zotero does not provide an endpoint that reveals remembered keys, so the key
must be copied from that application's private configuration; the server ID
alone is not enough to recover it.

BetterIssa's authorization store can be imported without printing the key:

```bash
zotero-mcp import-local-authorization \
  ~/.config/betterissa/zotero-local-authorizations.json
```

## Reconfigure or uninstall

Rerun the installer with new address/port options to replace the units.

```bash
sudo ./install.sh --listen-address 192.168.1.50 --listen-port 23120
sudo ./install.sh --uninstall
```

The proxy only forwards traffic. It does not modify Zotero's files or database.
