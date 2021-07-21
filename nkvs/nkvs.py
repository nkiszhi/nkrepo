#!/usr/bin/env python
#-*- encoding: utf-8 -*- 
# Nankai University Anti-Virus Group
# Zhi Wang zwang@nankai.edu.cn

from __future__ import print_function
import argparse
import requests
import re
import os

def get_html(usr, pwd):
    global html
    data = {    
        'username': usr,
        'password': pwd,
        }
    headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/73.0.3683.75 Chrome/73.0.3683.75 Safari/537.36',}
    login_url = 'https://virusshare.com/processlogin'
    session = requests.Session()
    resp = session.post(url=login_url,data=data,headers=headers)
    html_= resp.text
    with open("html_.txt","w") as f:
        f.write(html_)
    if resp.status_code != 200:
	#print(resp.status_code)
        print("[!]: Login failed!")
        print(resp.status_code)
        return
    else:
        print("[o]: Login successfully!")
    #print(resp.status_code)
    #print(resp.text)
    url = 'https://virusshare.com/torrents.4n6'
    resp = session.post(url=url,headers=headers)
    if resp.status_code != 200:
        print("[!]: Download HTML failed!")
        return
    else:
        print("[o]: Download HTML successfully!")
    html = resp.text
    #print(html)
    with open("html.txt","w") as f:
        f.write(html)
    return html

def check_update(html):
    """Check if there are new torrents and update url_list.txt."""

    URL_LIST_FILE = "url_list.txt"
    url_list = re.findall(r"<a.*?href=\"(.*VirusShare_.*)\">.*<\/a>",html,re.I)
    print("[o]: Found {} torrent files on VirusShare.com.".format(len(url_list)))
    if not os.path.exists(URL_LIST_FILE):
        print("[o]: {} new torrents.".format(len(url_list)))
        with open(URL_LIST_FILE, 'w+') as f:
            for item in url_list:
                f.write("%s\n" % item)
        return url_list
    else:
        with open(URL_LIST_FILE, 'r') as f:
            old_url_list = [line.rstrip('\n') for line in f]
        url_list = set(url_list) - set(old_url_list)
        if len(url_list) == 0:
            print("[!]: No update needed!")
        else:
            print("[o]: {} new torrents.".format(len(url_list)))
        with open(URL_LIST_FILE, 'a') as f:
            for item in url_list:
                f.write("%s\n" % item)
        return url_list

def download_torrent_file(url_list, torrent_dir):

    if not os.path.exists(torrent_dir):
        os.makedirs(torrent_dir)
    for u in url_list:
        u = u.strip()
        torrent_file = os.path.join(torrent_dir, u.split('?')[0].split('/')[-1])
        requests.adapters.DEFAULT_RETRIES = 5
        s = requests.session()
        s.keep_alive = False
        r = requests.get(u, verify=False)
        #print(r.status_code)
        if r.status_code == 200:
            print("[o]: Download {} successfully!".format(torrent_file))
            with open(torrent_file,'wb') as f:
                f.write(r.content)
        else:
            print("[!]: Download {} failed!".format(torrent_file))

def main():
    parser = argparse.ArgumentParser(prog="nkvs", description='Download shared malware samples from VirusShare.com.')
    parser.add_argument("-u", "--username", help="The username to login VirusShare.com")
    parser.add_argument("-p", "--password", help="The password to login VirusShare.com")
    parser.add_argument("-t", "--torrents", default="./DATA", help="The folder to store torrent files of VirusShare.com (default: ./DATA)")
    args = parser.parse_args()
    usr =  args.username
    pwd = args.password
    torrent_dir = args.torrents
    print("[o]: VirusShare.com user {}.".format(usr))
    #print(pwd)

    html = get_html(usr, pwd)
    url_list = check_update(html)
    download_torrent_file(url_list, torrent_dir)
    return

if __name__ == "__main__":
    main()

