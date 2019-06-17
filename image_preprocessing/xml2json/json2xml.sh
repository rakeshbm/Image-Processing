#!bin/sh

Folder_A="./"
for file_a in ${Folder_A}/*; do
    temp_file=`basename $file_a`
    filesubfix=${temp_file:0-5}
    if [ $filesubfix != ".json" ]
    then
        continue
    fi
    # echo $filesubfix
    substr=${temp_file%.*}
    xmlstr=$substr".xml"
    python json2xml.py -t json2xml -o $xmlstr $temp_file
    echo "JSON2XML sucess: "$substr".xml"
done
