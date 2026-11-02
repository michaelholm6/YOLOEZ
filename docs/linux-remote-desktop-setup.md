[< Back to README](../README.md)

# Linux Remote Desktop Setup

This section explains how to set up a lightweight Linux remote desktop using **XFCE** and **TightVNC**, with minimal use of `sudo`. The desktop can be accessed from **Windows using RealVNC Viewer**. This allows for use of YOLOEZ on Linux-based headless GPU clusters.

## Requirements

- Linux machine (Debian/Ubuntu-based)
- Non-root user account
- Network or SSH access
- Windows machine for remote access

## 1. Install Required Packages on Linux machine

```bash
sudo apt update
sudo apt install -y xfce4 xfce4-goodies tightvncserver unzip
```

Installed components:

* **xfce4**: lightweight desktop environment
* **xfce4-goodies**: additional XFCE utilities
* **tightvncserver**: VNC server
* **unzip**: utility to extract zip files

## 2. Download all parts from the [releases page](https://github.com/michaelholm6/YOLOEZ/releases) to your **local Windows machine** for your specific architecture. If you're not sure which files you need, you probably need the ones titled YOLOEZ-linux-x86_64.zip.001, YOLOEZ-linux-x86_64.zip.002, etc.

## 3. Copy the files to your Linux server using `scp` (or WinSCP):

```powershell
scp path/to/YOLOEZ-linux-x86_64.zip.* username@linux_host_ip:/home/username/
```

> Replace `username` and `linux_host_ip` with your Linux credentials. Replace ```path/to/``` with the path that you downloaded the individual zip files to.

## 4. SSH into the Linux server:

```bash
ssh username@linux_host_ip
```

## 5. Navigate to the folder containing the split ZIPs:

```bash
cd /home/username
```

## 6. Combine the split ZIP files into a single ZIP:

```bash
cat YOLOEZ-linux-x86_64.zip.* > YOLOEZ-linux-x86_64.zip
```

> `cat` concatenates the numeric parts in order (`.001`, `.002`, …). Make sure they are named sequentially.

## 7. Extract the combined ZIP:

```bash
unzip YOLOEZ-linux-x86_64.zip
```

## 8. Make the YOLOEZ executable runnable:

The archive extracts to a folder named `YOLOEZ-linux-x86_64`, which contains an executable of the same name alongside an `_internal` folder. The two must stay together.

```bash
cd YOLOEZ-linux-x86_64
chmod +x YOLOEZ-linux-x86_64
```

## 9. Initialize TightVNC

Run the VNC server once to set a password and create configuration files:

```bash
tightvncserver
```

After setup completes, stop the server:

```bash
tightvncserver -kill :1
```

## 10. Configure VNC to Start XFCE

```bash
nano ~/.vnc/xstartup
```

Replace contents with:

```sh
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

exec startxfce4 &
```

Make executable:

```bash
chmod +x ~/.vnc/xstartup
```

## 11. Start the VNC Server

```bash
tightvncserver -geometry 1920x1080
```

Example output:

```
New 'X' desktop is hostname:1
```

This means:

* Display `:1`
* Port `5901`


## 12. Firewall Configuration (optional)

If you need direct access and a firewall is enabled:

```bash
sudo ufw allow 5901/tcp
```

## 13. Connect Securely Using SSH Tunnel

From Windows PowerShell:

```powershell
ssh -L 5901:localhost:5901 username@linux_host_ip
```

This forwards the VNC connection securely over SSH.


## 14. Install RealVNC Viewer on Windows

Download and install **RealVNC Viewer** (Viewer only):

* [https://www.realvnc.com/en/connect/download/viewer/](https://www.realvnc.com/en/connect/download/viewer/)


## 15. Connect from Windows

1. Open **RealVNC Viewer**
2. Enter the connection address:

     ```
     localhost:5901
     ```
     
3. Click **Connect**
4. Enter your VNC password

You should now see the XFCE desktop.

## 16. Open Executable

1. In the remote desktop viewer, open the file manager and navigate to `/home/username/YOLOEZ-linux-x86_64`
2. Run the `YOLOEZ-linux-x86_64` executable inside that folder
3. The tool will now guide you through using it


## 17. Managing VNC Sessions

Stop a session:

```bash
tightvncserver -kill :1
```

Use `:2`, `:3`, and so on for additional sessions, depending on how many you are running.


## Notes

* Each display `:N` uses port `5900 + N`
* VNC passwords are separate from system passwords


