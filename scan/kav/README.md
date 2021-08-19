# Get Kaspersky scan results

Use Kaspersky anti-virus scan engine to scan all samples and save the
scan results in ".kav" file. Each sample has a corresponding ".kav"
file to save the scan result, which uses the sample's SHA256 value as
the file name. For example, the sample's SHA256 is 123456, so the scan
result is stored in the "12345.kav" file. 

## count_kav_label.py
This script could search the entire repo and count how many ".kav" files there are. 

## read_log.py

This script would read Kaspersky scan log file and save the scan
results into corresponding ".kav" files. This script uses regular
expression to search the log file and extract SHA256 values and the
corresponding detection results. 

## cp_unlabel_sample.py

This script would search entire sample repo to find out the samples
without the corresponding ".kav" file. Then this script copy such
samples into a temp folder for Kaspersky anti-virus engine to scan. 

## del_dup.py

This script would remove the samples which finished Kaspersky engine scan. The input is a txt file containing a list of SHA256 values. This script would delete the samples in the list from temp folder.


