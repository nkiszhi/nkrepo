# Synopsis
 Download malware samples from VirusShare.com

## Technical Freatures
- Download a VirusShare full torrent list
- Check for VirusShare updates
- Download torrent files
- Download zip files containing malware samples via transmission-daemon

## Usage

### Download torrent files from VirusShare.com

`python nkvs.py -u username -p password -t torrent_folder`

-u USERNAME, --username the username to login VirusShare.com

-p PASSWORD, the password to login VirusShare.com

-t TORRENTS_FOLDER, the folder to store torrent files

### Transmission-Daemon

Add aliases into .bashrc to create commands for torrent downloading.

1. Use command vsls to list all torrents

`alias vsls='transmission-remote -l'`

2. Use command vsinit to init transmission-daemon for downloading

`rm -fr ./config/transmission-daemon/torrents/*`

`alias vsinit='transmission-daemon --paused -w ./DATA -c ./DATA -e transmission.log -g ./config/transmission-daemon'`

First, clean all torrent files at transmision configration before init transmission for download.

-w --download-dir: directory to store downloaded data 

-c: directory to watch for new .torrent files to be added. 

3. Use command vsstart to start torrent downloading

`alias vsstart='transmission-remote --torrent all --start'`

4. Use command vsstop to stop torrent downloading

`alias vsstop='transmission-remote --torrent all --stop'`

5. Use command vskill to kill all transmission daemons

`alias vskill='pkill -f transmission'`

6. Change the download queue size. Open the file /etc/transmission-daemon/setting.json and change default value 5 of "download-queue-size" to 100. 

## Dependencies

`sudo apt-get install transmission-cli transmission-daemon`

nkvs更新流程

一．运行nkvs.py下载virusshare网站更新的种子文件

（nkvs.py的使用python nkvs.py -u username -p password）

nkvs.py中有三个函数，功能如下：

get_html(usr, pwd)登录virusshare网站，得到网页源码存于html中

check_update(html)解析html信息得到网站目前所有的种子文件存于url_list中，旧的已下载文件在url_list.txt中，将url_list所有种子文件减去list.txt中之前已下载的种子文件，得到更新未下载过的的种子文件存于url_list

download_torrent_file(url_list, torrent_dir)下载网站更新的种子文件到./DATA中
（注：不同时候的网站爬虫网页URL可能会不同，在运行nkvs.py出错时可以先核对网页URL是否正确）

二．将下载的种子文件使用命令下载对应的zip文件

1. 运行kill命令把所有的transmission进程关掉（ps aux |grep transmission列出所有的transmission进程信息）（kill -9 进程号关闭对应进程）

2. 进入/home/RaidDisk/nkvs路径（必须在该路径下运行vsinit命令）

3. 运行vsinit，启动transmission进程

4. 运行vsstart，启动zip文件下载

5. 运行vsls，查看zip文件的下载进度

三．所有zip文件下载好后，运行vs_unzip.py

vs_unzip.py：解压缩从virusshare下载下来的zip文件，并移动到nkrepo对应文件夹下


