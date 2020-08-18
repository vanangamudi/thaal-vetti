
COMMAND=$1 #csm
INPUT=$2

INPUT_PAGES_DIR=${INPUT}_pages
INPUT_SPLIT_PAGES_DIR=${INPUT}_split_pages

mkdir -p $INPUT_PAGES_DIR $INPUT_SPLIT_PAGES_DIR

if [[ $COMMAND == *"c"* ]]
then
    echo "chunking pdf file into pages"
    # remove PNG24: and add necessary info into problems and fixes file
    # convert -set colorspace RGB -density 300 $INPUT PNG24:$INPUT_PAGES_DIR/%04d.png
    gs  -dBATCH -dNOPAUSE -dSAFER -sDEVICE=png16m  -dJPEGQ=95 -r300x300 -sOutputFile=$INPUT_PAGES_DIR/%04d.png  $INPUT
    
    RET=$?
    if test $RET -ne 0
    then
	echo "ERROR:${RET} chunking failed!!!"
	
	mkdir -p errored
	
	rm -rd errored/$INPUT_PAGES_DIR
	mv $INPUT_PAGES_DIR errored
	
	exit -1
    fi

fi

if [[ $COMMAND == *"s"* ]]
then

    # per page processing, in this case just copying the pages
    echo "doing the thing"
    #cp $INPUT_PAGES_DIR/* $INPUT_SPLIT_PAGES_DIR
    python3 split.py -i $INPUT_PAGES_DIR -o $INPUT_SPLIT_PAGES_DIR
    
    RET=$?
    if test $RET -ne 0
    then
	echo "ERROR:${RET} spliting failed!!!"
	
	mkdir -p errored
	
	rm -rd errored/$INPUT_PAGES_DIR
	mv $INPUT_PAGES_DIR  errored
	
	rm -rd errored/$INPUT_SPLIT_PAGES_DIR
	mv $INPUT_SPLIT_PAGES_DIR errored
	
	exit -1
    fi
    
fi


if [[ $COMMAND == *"m"* ]]
then

    echo "merging the pages into pdf"
    convert $INPUT_SPLIT_PAGES_DIR/*.png  ${INPUT}_split.pdf
    
    
    RET=$?
    if test $RET -ne 0
    then
	echo "ERROR:${RET} spliting failed!!!"
	
	mkdir -p errored
	
	rm -rd errored/$INPUT_PAGES_DIR
	mv $INPUT_PAGES_DIR  errored
	
	rm -rd errored/$INPUT_SPLIT_PAGES_DIR
	mv $INPUT_SPLIT_PAGES_DIR errored
	
	exit -1
    fi

    
    rm -rf $INPUT_PAGES_DIR $INPUT_SPLIT_PAGES_DIR


fi




########
# If you get the following error
#
# convert-im6.q16: attempt to perform an operation not allowed by the security policy `PDF' @ error/constitute.c/IsCoderAuthorized/408.
#
# paste this in imagemagick config
#
# >>   <policy domain="coder" rights="read | write" pattern="PDF" />
# in policymap section, i.e just before </policymap> in '/etc/ImageMagick-7/policy.xml'
#
#    reference: https://stackoverflow.com/questions/52998331/imagemagick-security-policy-pdf-blocking-conversion
