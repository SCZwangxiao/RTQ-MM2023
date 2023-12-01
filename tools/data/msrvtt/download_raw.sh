# Download MSRVTT videos and annotations for Retrieval
DOWNLOAD=${1:-"."}

for FOLDER in 'vis_db' 'txt_db'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://convaisharables.blob.core.windows.net/clipbert'

# vis dbs
if [ ! -d $DOWNLOAD/vis_db/msrvtt/ ] ; then
    wget -nc $BLOB/vis_db/msrvtt.tar -P $DOWNLOAD/vis_db/
    mkdir -p $DOWNLOAD/vis_db/msrvtt
    tar -xvf $DOWNLOAD/vis_db/msrvtt.tar -C $DOWNLOAD/vis_db/msrvtt
    rm $DOWNLOAD/vis_db/msrvtt.tar
fi

# text dbs
if [ ! -d $DOWNLOAD/txt_db/msrvtt_retrieval/ ] ; then
    # MC-Test is included
    wget -nc $BLOB/txt_db/msrvtt_retrieval.tar -P $DOWNLOAD/txt_db/
    mkdir -p $DOWNLOAD/txt_db/msrvtt_retrieval
    tar -xvf $DOWNLOAD/txt_db/msrvtt_retrieval.tar -C $DOWNLOAD/txt_db/msrvtt_retrieval
    rm $DOWNLOAD/txt_db/msrvtt_retrieval.tar
fi


# Download MSRVTT caption annotations
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt/cap_train.json -P $DOWNLOAD/txt_db/msrvtt_caption
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt/cap_val.json -P $DOWNLOAD/txt_db/msrvtt_caption
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt/cap_test.json -P $DOWNLOAD/txt_db/msrvtt_caption


# Download MSRVTT QA annotations
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt/qa_train.json -P $DOWNLOAD/txt_db/msrvtt_qa
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt/qa_val.json -P $DOWNLOAD/txt_db/msrvtt_qa
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/msrvtt/qa_test.json -P $DOWNLOAD/txt_db/msrvtt_qa