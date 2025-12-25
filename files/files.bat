@echo off
:: Initialize/Clear the master file
type nul > files.txt

echo Assembling files...

:: Explicit Folder Strikes
type 0001-0010\*.* >> files.txt
type 0011-0020\*.* >> files.txt
type 0021-0030\*.* >> files.txt
type 0031-0040\*.* >> files.txt
type 0041-0050\*.* >> files.txt
type 0051-0060\*.* >> files.txt
type 0061-0070\*.* >> files.txt
type 0071-0080\*.* >> files.txt
type 0081-0090\*.* >> files.txt

echo Assembly complete.
