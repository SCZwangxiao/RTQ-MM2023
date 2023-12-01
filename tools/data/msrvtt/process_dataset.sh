DOWNLOAD=${1:-"."}

cp $DOWNLOAD/txt_db/msrvtt_retrieval/train.jsonl $DOWNLOAD/txt_db/msrvtt_retrieval/train_1kA.jsonl
cat '\n' >> $DOWNLOAD/txt_db/msrvtt_retrieval/train_1kA.jsonl
cat $DOWNLOAD/txt_db/msrvtt_retrieval/val.jsonl >> $DOWNLOAD/txt_db/msrvtt_retrieval/train_1kA.jsonl

python convert_data_format.py $DOWNLOAD/txt_db/msrvtt_caption
cp $DOWNLOAD/txt_db/msrvtt_caption/train.jsonl $DOWNLOAD/txt_db/msrvtt_caption/train_original.jsonl
cat $DOWNLOAD/txt_db/msrvtt_caption/val.jsonl >> $DOWNLOAD/txt_db/msrvtt_caption/train.jsonl
