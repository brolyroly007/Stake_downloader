from recorders.clip_downloader import ClipDownloader
print("Attributes of ClipDownloader:")
print([x for x in dir(ClipDownloader) if not x.startswith("_")])
