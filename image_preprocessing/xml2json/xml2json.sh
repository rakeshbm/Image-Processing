#!bin/sh

Folder_A="./"
for file_a in ${Folder_A}/*; do
    temp_file=`basename $file_a`
    filesubfix=${temp_file:0-4}
    if [ $filesubfix != ".xml" ]
    then
    	continue
    fi
    # echo $filesubfix
    substr=${temp_file%.*}
    jsonstr=$substr".json"
    xml2json -t xml2json -o $jsonstr $temp_file --strip_text
    echo "XML2JSON sucess: "$substr".json"
done