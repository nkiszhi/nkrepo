# Synopsis
 Download malware samples from VirusShare.com

## Usage

vs\_unzip.py: extract samples from virusshare zip files.

vs\_mov.py: move samples into nkrepo.

### P2P tool: Transmission-Daemon

#### Installation

`sudo apt-get install transmission-cli transmission-daemon`

#### Use command vsls to list all torrents

Add aliases into .bashrc to create commands for torrent downloading.

`alias vsls='transmission-remote -l'`

#### Use command vsinit to init transmission-daemon for downloading

`rm -fr ./config/transmission-daemon/torrents/*`

`alias vsinit='transmission-daemon --paused -w ./DATA -c ./DATA -e transmission.log -g ./config/transmission-daemon'`

First, clean all torrent files at transmision configration before init transmission for download.

-w --download-dir: directory to store downloaded data 

-c: directory to watch for new .torrent files to be added. 

#### Use command vsstart to start torrent downloading

`alias vsstart='transmission-remote --torrent all --start'`

#### Use command vsstop to stop torrent downloading

`alias vsstop='transmission-remote --torrent all --stop'`

#### Use command vskill to kill all transmission daemons

`alias vskill='pkill -f transmission'`

#### Change the download queue size. 

Open the file /etc/transmission-daemon/setting.json and change default value 5 of "download-queue-size" to 100. 


