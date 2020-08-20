#### configs
# Chunking config
PAGE_RESOLUTION=150

# vetti.py scale factor, if the window appears too big decrease this number
DISPLAY_RESOLUTION=0.5
DEBUG='' # for debugging make it '--debug'
CLEAR='' # for clearing old state make it '--clear',
           #if you change display resolution set this to '--clear'


#### end of configs ####


COMMAND=$1 #csm similar to chmod rwx
INPUT=$2

INPUT_PAGES_DIR=${INPUT}_pages
INPUT_SPLIT_PAGES_DIR=${INPUT}_split_pages

mkdir -p $INPUT_PAGES_DIR

if [[ $COMMAND == *"c"* ]]
then
    echo "chunking pdf file into pages"
    # remove PNG24: and add necessary info into problems and fixes file
    # convert -set colorspace RGB -density 300 $INPUT PNG24:$INPUT_PAGES_DIR/%04d.png
    npages=$(pdfinfo $INPUT | grep Pages: | cut -d ':' -f 2 | xargs)
    
    gs  -dBATCH \
	-dNOPAUSE \
	-dSAFER \
	-sDEVICE=png16m  \
	-dJPEGQ=95 \
	-r${PAGE_RESOLUTION}x${PAGE_RESOLUTION} \
	-sOutputFile=$INPUT_PAGES_DIR/%04d_$npages.png \
	$INPUT
    
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
    mkdir -p  $INPUT_SPLIT_PAGES_DIR
    #cp $INPUT_PAGES_DIR/* $INPUT_SPLIT_PAGES_DIR
    for filename in $INPUT_PAGES_DIR/*.png;
    do
	python3 vetti.py \
		--display-resolution $DISPLAY_RESOLUTION \
		$DEBUG \
		$CLEAR \
		-i $filename \
		-o $INPUT_SPLIT_PAGES_DIR

	
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
    done
    
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


