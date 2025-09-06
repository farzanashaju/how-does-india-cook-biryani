# to download all youtube videos, with transcripts and metadata

import yt_dlp
import os
import json
import logging
import re
import glob
import shutil
import paramiko
import subprocess

# config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RESOLUTION = '720'

SUB_EXTS = ['.vtt', '.srt', '.ttml', '.json']

BIRYANI_URLS = {
        'ambur_biryani': 
            ['https://www.youtube.com/watch?v=bQ_OUze0bhU',
            'https://www.youtube.com/watch?v=ajbVOfXT8ME',
            'https://www.youtube.com/watch?v=rmWcuTdVPvc',
            'https://www.youtube.com/watch?v=G1-I_qvKRbY',
            'https://www.youtube.com/watch?v=MPPyNaxS_BM',
            'https://www.youtube.com/watch?v=TphED4WaZbo',
            'https://www.youtube.com/watch?v=aF-NYduPBaA',
            'https://www.youtube.com/watch?v=ktwqpScthdA',
            'https://www.youtube.com/watch?v=DorxTA0igIE',
            'https://www.youtube.com/watch?v=VlJgJN-ihyA'],
    'bombay_biryani': 
            ['https://www.youtube.com/watch?v=DBpxjLbY9Sg',
            'https://www.youtube.com/watch?v=_v2aMYU16W4',
            'https://www.youtube.com/watch?v=dIieqJP1mac',
            'https://www.youtube.com/watch?v=yxlJEImuyYU',
            'https://www.youtube.com/watch?v=-yZFPs8jqFw',
            'https://www.youtube.com/watch?v=ntHJLC1rAz0',
            'https://www.youtube.com/watch?v=QTMffxDiTeM',
            'https://www.youtube.com/watch?v=L0ummwp0pJA',
            'https://www.youtube.com/watch?v=kpJaaEvPR_U',
            'https://www.youtube.com/watch?v=DBpxjLbY9Sg'],
    'dindigul_biryani': 
            ['https://www.youtube.com/watch?v=5Zra4nFepRg',
            'https://www.youtube.com/watch?v=gAz0tchMS6M',
            'https://www.youtube.com/watch?v=1GB9IMAaMFA',
            'https://www.youtube.com/watch?v=eWXHPRvfZTY',
            'https://www.youtube.com/watch?v=iARLgsvABRY',
            'https://www.youtube.com/watch?v=HYZz6K78iG4',
            'https://www.youtube.com/watch?v=AgbGZg5nVr4',
            'https://www.youtube.com/watch?v=pRRioas9ZX4',
            'https://www.youtube.com/watch?v=d3PIOllHsYE',
            'https://www.youtube.com/watch?v=hgI4wV_WoVs'],
    'donne_biryani': 
            ['https://www.youtube.com/watch?v=DMXlJY-3uA4',
            'https://www.youtube.com/watch?v=FB8jiqqOK68',
            'https://www.youtube.com/watch?v=YdpAYLIjz_4',
            'https://www.youtube.com/watch?v=-DoPE08CvUI',
            'https://www.youtube.com/watch?v=p2esRosqD8U',
            'https://www.youtube.com/watch?v=9o0lljXKAPs',
            'https://www.youtube.com/watch?v=ErDQKcqC1dQ',
            'https://www.youtube.com/watch?v=iBIu31ZJ4jU',
            'https://www.youtube.com/watch?v=l0n4eHBzPbk',
            'https://www.youtube.com/watch?v=Gj6gxysrkh8'],
    'hyderabadi_biryani': 
            ['https://www.youtube.com/watch?v=uXf3xXeu1x4',
            'https://www.youtube.com/watch?v=nf9tq7cNkTQ',
            'https://www.youtube.com/watch?v=zkAxXHSHpv0',
            'https://www.youtube.com/watch?v=0DE-2vniskw',
            'https://www.youtube.com/watch?v=BIXMwLFCboA',
            'https://www.youtube.com/watch?v=FWKNGUeIX7w',
            'https://www.youtube.com/watch?v=jCABb8HsrWQ',
            'https://www.youtube.com/watch?v=v-dgOnlnR40',
            'https://www.youtube.com/watch?v=yAasjcAEOvo',
            'https://www.youtube.com/watch?v=E_gWBBjYkjE'],
    'kashmiri_biryani': 
            ['https://www.youtube.com/watch?v=oZW6IaRTcX8',
            'https://www.youtube.com/watch?v=9NDYjA2G0yo',
            'https://www.youtube.com/watch?v=8xIteGN3FfU',
            'https://www.youtube.com/watch?v=PX2oRDCLF80',
            'https://www.youtube.com/watch?v=S0Ow7Lmu7Pg',
            'https://www.youtube.com/watch?v=1xOXRMUjGjQ',
            'https://www.youtube.com/watch?v=iZa2pttrdLc',
            'https://www.youtube.com/watch?v=cxWnqns977g',
            'https://www.youtube.com/watch?v=5N2mb1xyUCI',
            'https://www.youtube.com/watch?v=1yHrE_V5gpA'],
    'kolkata_biryani': 
            ['https://www.youtube.com/watch?v=cewMQvUAESI',
            'https://www.youtube.com/watch?v=9EgcWzkt5Zk',
            'https://www.youtube.com/watch?v=XplW4n75UrI',
            'https://www.youtube.com/watch?v=jtXPYRmiBbI',
            'https://www.youtube.com/watch?v=4tDu8V26i9g',
            'https://www.youtube.com/watch?v=YBd1TcCfWBs',
            'https://www.youtube.com/watch?v=1PjtS2fr-xI',
            'https://www.youtube.com/watch?v=8P8vdhEp8OA',
            'https://www.youtube.com/watch?v=2ahDWniP4YY',
            'https://www.youtube.com/watch?v=70RNVsBEr44'],
    'lucknow_awadhi_biryani': 
            ['https://www.youtube.com/watch?v=8BT91Oxp_C0',
            'https://www.youtube.com/watch?v=mtI-2Z2ZC-I',
            'https://www.youtube.com/watch?v=dbQnkRRqp5E',
            'https://www.youtube.com/watch?v=ZKjAIyb6UUw',
            'https://www.youtube.com/watch?v=B16qGzPTgPM',
            'https://www.youtube.com/watch?v=McP-YCUbJl4',
            'https://www.youtube.com/watch?v=6iuuetZ9YaE',
            'https://www.youtube.com/watch?v=oVEEZ-cBxeI',
            'https://www.youtube.com/watch?v=kNnn3XqLWFA',
            'https://www.youtube.com/watch?v=SXI-U24Bs4o'],
    'malabar_biryani': 
            ['https://www.youtube.com/watch?v=_Jk-YuzqnaY',
            'https://www.youtube.com/watch?v=zTR1bL2wv1o',
            'https://www.youtube.com/watch?v=emZ4Rn9kgwY',
            'https://www.youtube.com/watch?v=4THANac_Rhs',
            'https://www.youtube.com/watch?v=yV3JAFDqR0c',
            'https://www.youtube.com/watch?v=PEDwxukasLc',
            'https://www.youtube.com/watch?v=FU5VMxP1o3w',
            'https://www.youtube.com/watch?v=oyVBoqbAdYI',
            'https://www.youtube.com/watch?v=3afTviMrHN0',
            'https://www.youtube.com/watch?v=hEF3YfPmYcM'],
    'mughlai_biryani': 
            ['https://www.youtube.com/watch?v=QWjgN52whDQ',
            'https://www.youtube.com/watch?v=x63VozM-6SY',
            'https://www.youtube.com/watch?v=Nrh6A4LD4Bw',
            'https://www.youtube.com/watch?v=-XlNmh4x4tg',
            'https://www.youtube.com/watch?v=PwVbvj8tlUE',
            'https://www.youtube.com/watch?v=JTIQSyvYEfg',
            'https://www.youtube.com/watch?v=YGuVtWRO_14',
            'https://www.youtube.com/watch?v=GZdZEExzKu8',
            'https://www.youtube.com/watch?v=i2WGTjEaRbg',
            'https://www.youtube.com/watch?v=80lt5q4HTuE'],
    'sindhi_biryani': 
            ['https://www.youtube.com/watch?v=exu4IuxgNwU',
            'https://www.youtube.com/watch?v=otNb4UI60dU',
            'https://www.youtube.com/watch?v=iQ5mDJjaP2k',
            'https://www.youtube.com/watch?v=SEGSmBn7w6U',
            'https://www.youtube.com/watch?v=19e0rLz4MwA',
            'https://www.youtube.com/watch?v=QMovteZpDKI',
            'https://www.youtube.com/watch?v=mivJuWYfcYY',
            'https://www.youtube.com/watch?v=vRXY8xdTBP0',
            'https://www.youtube.com/watch?v=h9mugWdOoo4',
            'https://www.youtube.com/watch?v=GezHL0ti26M'],
    'thalassery_biryani': 
            ['https://www.youtube.com/watch?v=fw_z1nxkWGI',
            'https://www.youtube.com/watch?v=OSDIUUjnFdk',
            'https://www.youtube.com/watch?v=VPKDh8jdLQ4',
            'https://www.youtube.com/watch?v=48uQBMbvTXg',
            'https://www.youtube.com/watch?v=DZbWuOUyQ5k',
            'https://www.youtube.com/watch?v=CX5t8RX8ct0',
            'https://www.youtube.com/watch?v=o-J-h8jsFU8',
            'https://www.youtube.com/watch?v=F4rTfLw3t58',
            'https://www.youtube.com/watch?v=agmuPe_Osbg',
            'https://www.youtube.com/watch?v=bk_QzacJM68']
}

# utilities

def rename_transcripts(directory):
    for ext in SUB_EXTS:
        for old_path in glob.glob(os.path.join(directory, f"*{ext}")):
            filename = os.path.basename(old_path)
            if filename == 'metadata.json':
                continue
            match = re.search(r'\.([a-zA-Z]{2}(?:-[a-zA-Z]{2})?)\.(vtt|srt|ttml|json)$', filename)
            if match:
                lang_code = match.group(1)
                new_filename = f"transcript_{lang_code}.{match.group(2)}"
            elif 'auto' in filename.lower():
                new_filename = f"transcript_auto.{ext.lstrip('.')}"
            else:
                new_filename = f"transcript_unknown.{ext.lstrip('.')}"
            new_path = os.path.join(directory, new_filename)
            counter = 1
            while os.path.exists(new_path) and new_path != old_path:
                base, extn = os.path.splitext(new_filename)
                new_filename = f"{base}_{counter}{extn}"
                new_path = os.path.join(directory, new_filename)
                counter += 1
            os.rename(old_path, new_path)

def rename_audio_file(directory):
    video_file = os.path.join(directory, 'video.mp4')
    audio_file = os.path.join(directory, 'audio.mp3')
    if os.path.exists(video_file) and not os.path.exists(audio_file):
        try:
            subprocess.run(['ffmpeg', '-i', video_file, '-vn', '-acodec', 'mp3', '-ab', '192k', '-y', audio_file],
                           check=True, capture_output=True)
        except Exception as e:
            print(f"Audio extraction error: {e}")

def upload_folder_to_server(local_path, remote_path):
    transport = paramiko.Transport((REMOTE_HOST, 22))
    transport.connect(username=REMOTE_USERNAME, password=REMOTE_PASSWORD)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # create remote directories
    parts = remote_path.strip('/').split('/')
    current = ''
    for part in parts:
        current += '/' + part
        try:
            sftp.mkdir(current)
        except IOError:
            pass

    # upload files
    for root, dirs, files in os.walk(local_path):
        rel_path = os.path.relpath(root, local_path)
        remote_root = os.path.join(remote_path, rel_path).replace("\\", "/")
        try:
            sftp.mkdir(remote_root)
        except:
            pass
        for file in files:
            sftp.put(os.path.join(root, file), os.path.join(remote_root, file).replace("\\", "/"))

    sftp.close()
    transport.close()

# main pipeline

def download_biryani_videos():
    for biryani_type, urls in BIRYANI_URLS.items():
        biryani_dir = os.path.join(TEMP_PATH, biryani_type)
        os.makedirs(biryani_dir, exist_ok=True)

        for idx, url in enumerate(urls, 1):
            video_dir = os.path.join(biryani_dir, f"video{idx}")
            os.makedirs(video_dir, exist_ok=True)

            try:
                ydl_opts_download = {
                    'outtmpl': os.path.join(video_dir, 'video.%(ext)s'),
                    'format': f'(bestvideo[height<={RESOLUTION}]+bestaudio)/(best)',
                    'merge_output_format': 'mp4',
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en', 'hi', 'ta', 'te', 'ml', 'bn', 'gu', 'kn', 'mr', 'pa'],
                    'subtitlesformat': 'best',
                    'quiet': True,
                    'no_warnings': True,
                    'prefer_ffmpeg': True,
                    'postprocessors': [],
                }

                with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
                    info = ydl.extract_info(url, download=True)

                # save metadata
                metadata = {
                    "title": info['title'],
                    "url": url,
                    "id": info['id'],
                    "uploader": info.get('uploader'),
                    "dish": biryani_type
                }
                with open(os.path.join(video_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)

                rename_audio_file(video_dir)
                rename_transcripts(video_dir)

                # upload
                print(f"✓ Downloaded video {biryani_type} #{idx}: {info['title']}")
                remote_path = os.path.join(REMOTE_BASE_PATH, biryani_type, f"video{idx}")
                upload_folder_to_server(video_dir, remote_path)

            except Exception as e:
                print(f"✗ Error for {biryani_type} #{idx}: {e}")
                shutil.rmtree(video_dir)

if __name__ == '__main__':
    download_biryani_videos()