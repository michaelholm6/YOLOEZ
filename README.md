# YOLOEZ

![CI main](https://github.com/michaelholm6/YOLOEZ/actions/workflows/ci.yml/badge.svg?branch=main)
![CI dev](https://github.com/michaelholm6/YOLOEZ/actions/workflows/ci.yml/badge.svg?branch=development)

A standalone, GUI based application for labeling data, training models, and running inference with Ultralytics powered YOLO11 models.  
This tool supports both bounding box detection and segmentation workflows and is designed to be usable without writing any code.

The application guides users through every step with built-in tooltips, instructional popups, and clear workflow structure, making YOLO model training and usage accessible to users who may not be familiar with machine learning pipelines or Python development.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Requirements](#system-requirements)
4. [Supported Tasks](#supported-tasks)
5. [Getting Started](#getting-started)
6. [Application Workflow](#application-workflow)
7. [User Guidance and Help System](#user-guidance-and-help-system)
8. [Linux Remote Desktop Setup](#linux-remote-desktop-setup)
9. [Screenshots and Visual Examples](#screenshots-and-visual-examples)
10. [Repository Structure](#repository-structure)
11. [Intended Audience](#intended-audience)
12. [License](#license)

---

## Overview

This repository contains the full source code and prebuilt executables for a graphical application that simplifies the process of working with YOLO11 models.

The tool provides an end-to-end workflow including:
- Dataset labeling
- Model training
- Model inference
- Visualization of results

All functionality is exposed through an intuitive graphical interface. No scripting or manual configuration is required.

---

## Key Features

- GUI driven YOLO11 training and inference
- Support for bounding box detection and segmentation models
- Integrated dataset labeling tools
- built-in training configuration interface
- Step by step instructional popups
- Context sensitive tooltips throughout the interface
- No coding required
- Packaged executables for easy installation

---

## System Requirements

- Windows or Linux (When running on a headless Linux server, a remote desktop must be used to interact with the GUI; terminal commands are outside the intended workflow.).
   - [Instructions for setting up remote desktop on a Linux server](#linux-remote-desktop-setup)
- GPU recommended for training but not required
- Sufficient disk space for datasets and trained models

Exact requirements may vary depending on dataset size and model configuration.

---

## Supported Tasks

- Image annotation using bounding boxes
- Image annotation using segmentation masks
- Dataset organization
- Training YOLO11 detection models
- Training YOLO11 segmentation models
- Running inference on new images
- Visualizing predictions directly in the GUI

---

## Getting Started

No installation or environment setup is required. If you're trying to run this on a linux remote desktop, reference [this section](#linux-remote-desktop-setup).

1. Navigate to the [**Releases**](https://github.com/michaelholm6/YOLOEZ/releases) section of this repository, located on the right side of the GitHub page.
2. Download the latest release for your operating system. If downloading for Linux, download all numbered zip files.
   NOTE: If downloading for Linux, refer to [this section](#6-combine-the-split-zip-files-into-a-single-zip) for guidance on combining multiple zip files.
3. Extract the downloaded zip file.
4. Launch the executable included in the extracted folder.
5. NOTE: You must keep the executable file in the same directory as the _internal folder.

The application will start immediately and guide you through the available workflows.

---

## Application Workflow

The application is organized into clear, sequential workflows:

1. **Labeling**
   - Load image datasets
   - Annotate images using built-in tools
   - Save labels in YOLO compatible format

2. **Training**
   - Configure training parameters through the GUI
   - Start training with a single click
   - Monitor progress within the application

3. **Inference**
   - Load a trained model
   - Run predictions on new images
   - Save results both visually and in JSON format

Each stage includes guidance to help users understand what is required before moving forward.

---

## User Guidance and Help System

The GUI is designed to be self-explanatory and instructional.

- Tooltips appear when hovering over blue question mark icons
- Instructional popups explain each step of a workflow
- Validation messages help prevent common mistakes
- Clear prompts guide users through required actions

This ensures that even first time users can successfully train and use YOLO models.

---

## Linux Remote Desktop Setup

This section explains how to set up a lightweight Linux remote desktop using **XFCE** and **TightVNC**, with minimal use of `sudo`. The desktop can be accessed from **Windows using RealVNC Viewer**. This allows for use of YOLOEZ on Linux-based headless GPU clusters.

### Requirements

- Linux machine (Debian/Ubuntu-based)
- Non-root user account
- Network or SSH access
- Windows machine for remote access

### 1. Install Required Packages on Linux machine

```bash
sudo apt update
sudo apt install -y xfce4 xfce4-goodies tightvncserver unzip
```

Installed components:

* **xfce4** — lightweight desktop environment
* **xfce4-goodies** — additional XFCE utilities
* **tightvncserver** — VNC server
* **unzip** — utility to extract zip files

### 2. Download all parts from the [releases page](https://github.com/michaelholm6/YOLOEZ/releases) to your **local Windows machine** for your specific architecture. If you're not sure which files you need, you probably need the ones title YOLOEZ-linux-x86_64.zip.001, YOLOEZ-linux-x86_64.zip.002, etc.

### 3. Copy the files to your Linux server using `scp` (or WinSCP):

```powershell
scp path/to/YOLOEZ-linux-x86_64.zip.* username@linux_host_ip:/home/username/
```

> Replace `username` and `linux_host_ip` with your Linux credentials. Replace ```path/to/``` with the path that you downloaded the individual zip files to.

### 4. SSH into the Linux server:

```bash
ssh username@linux_host_ip
```

### 5. Navigate to the folder containing the split ZIPs:

```bash
cd /home/username
```

### 6. Combine the split ZIP files into a single ZIP:

```bash
cat YOLOEZ-linux-x86_64.zip.* > YOLOEZ-linux-x86_64.zip
```

> `cat` concatenates the numeric parts in order (`.001`, `.002`, …). Make sure they are named sequentially.

### 7. Extract the combined ZIP:

```bash
unzip YOLOEZ-linux-x86_64.zip
```

### 8. Make the YOLOEZ executable runnable:

```bash
chmod +x YOLOEZ
```

### 9. Initialize TightVNC

Run the VNC server once to set a password and create configuration files:

```bash
tightvncserver
```

After setup completes, stop the server:

```bash
tightvncserver -kill :1
```

### 10. Configure VNC to Start XFCE

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

### 11. Start the VNC Server

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


### 12. Firewall Configuration (optional)

If you need direct access and a firewall is enabled:

```bash
sudo ufw allow 5901/tcp
```

### 13. Connect Securely Using SSH Tunnel

From Windows PowerShell:

```powershell
ssh -L 5901:localhost:5901 username@linux_host_ip
```

This forwards the VNC connection securely over SSH.


### 14. Install RealVNC Viewer on Windows

Download and install **RealVNC Viewer** (Viewer only):

* [https://www.realvnc.com/en/connect/download/viewer/](https://www.realvnc.com/en/connect/download/viewer/)


### 15. Connect from Windows

1. Open **RealVNC Viewer**
2. Enter the connection address:

     ```
     localhost:5901
     ```
     
3. Click **Connect**
4. Enter your VNC password

You should now see the XFCE desktop.

### 16. Open Executable

1. In the remote desktop viewer, navigate to /home/username
2. Find the unzipped executable, and run it
3. The tool will now guide you through using it


### 17. Managing VNC Sessions

List sessions:

```bash
tightvncserver -list
```

Stop a session:

```bash
tightvncserver -kill :1
```

Start a new session:

```bash
tightvncserver
```


### Notes

* Each display `:N` uses port `5900 + N`
* VNC passwords are separate from system passwords




## Screenshots and Visual Examples

### Main Application Window
![Main GUI Window](images/gui_main.png)

### Bounding Box Annotation Example
![Bounding Box Example](images/bounding_box.png)

### Segmentation Mask Example
![Segmentation Example](images/segmentation.png)

### Tooltip Example
![Tooltip Example](images/tooltip.png)

### Training Performance Panel
![Training Panel](images/training_panel.png)

### Inference Preparation View
![Inference Results](images/inference_page.png)

---

## Repository Structure
<pre>
├── src/                 # Application source code  
├── assets/              # Icons and UI assets  
├── docs/                # Documentation resources  
├── images/              # README images and screenshots  
├── compiler.py          # Script to automatically compile executable  
├── Various UV environment files  
└── README.md
</pre>

---

## Intended Audience

This tool is intended for:
- Researchers
- Students
- Engineers
- Domain experts without ML backgrounds
- Anyone who wants to train and use YOLO11 models without writing code

---

## License

This project is licensed under the AGPL-3.0 License – see the [LICENSE](LICENSE) file for details.
