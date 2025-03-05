
model=borzoi
for fold in {0,1,2,3,4}
do
	torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py -f \
	      -m $model  -d  $fold -v
 	torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
	 	-m $model  -d $fold -v
done