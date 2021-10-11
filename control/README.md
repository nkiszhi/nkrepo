# NKAMG Malware Dataset Control Components

## Count samples
count.py is used to count the number of files with specified file type. The
supported file types include sha256, md5, vt(VirusTotal scan results),
kav(Kaspersky scan results) and nolabel(files without vt nor kav results)

## Add new samples
add_sample.py is used to add samples into repository. The default folder containing new samples is TEMP.

## Delete samples
del_sample.py is used to delete samples based on file SHA256 values.

