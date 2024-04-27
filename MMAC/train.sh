

python train.py --lesion LC --model unet++ --opti adamw 
python train.py --lesion CNV --model unet++ --opti adamw 
python train.py --lesion FS --model unet++ --opti adamw 

python train.py --lesion LC --model manet --opti adamw 
python train.py --lesion CNV --model manet --opti adamw 
python train.py --lesion FS --model manet --opti adamw 

python train.py --lesion LC --model deeplabv3+ --opti adamw 
python train.py --lesion CNV --model deeplabv3+ --opti adamw 
python train.py --lesion FS --model deeplabv3+ --opti adamw 
