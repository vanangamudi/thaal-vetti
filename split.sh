
INPUT=$1

INPUT_PAGES_DIR=${INPUT}_pages
INPUT_SPLIT_PAGES_DIR=${INPUT}_split_pages

mkdir $INPUT_PAGES_DIR $INPUT_SPLIT_PAGES_DIR

echo "chunking pdf file into pages"
convert -set colorspace RGB -density 300 $INPUT PNG24:$INPUT_PAGES_DIR/%04d.jpg

# per page processing, in this case just copying the pages
echo "doing the thing"
cp $INPUT_PAGES_DIR/* $INPUT_SPLIT_PAGES_DIR

echo "merging the pages into pdf"
convert $INPUT_SPLIT_PAGES_DIR/*.jpg  ${INPUT}_split.pdf

rm -rf $INPUT_PAGES_DIR $INPUT_SPLIT_PAGES_DIR


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
