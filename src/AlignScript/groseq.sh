
cl="$1"
if [ -z "$cl" ]; then
    echo "Error: No cell line provided."
    exit 1
fi

echo "Cell line is $cl"


for i in *_pass.fastq.gz
do
    name=$(echo $i | awk -F"/" '{print $NF}' | awk -F"_pass.fastq.gz" '{print $1}')
    echo $name
    if [[ "$cl" == "K562" ]] || [[ "$cl" == "GM12878" ]]; then
        echo "Running the script for cell line $cl"
        cutadapt -m 5 -a TGGAATTCTCGG -o ${name}_trim_pass.fastq.gz ${name}_pass.fastq.gz
	      name=${name}_trim
	      echo $name
    else
        echo "pass"
    fi

    bowtie2 -p 3 -x /nfs/turbo/umms-drjieliu/proj/bowtie2_index/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
          -U ${name}_pass.fastq.gz | \
         samtools view -b - | \
         samtools sort - -o ${name}.sorted.bam
    module load openjdk/18.0.1.1
    java -jar picard.jar  \
        MarkDuplicates I=${name}.sorted.bam \
        O=${name}.sorted.marked.bam METRICS_FILE=${name}.sorted.marked.metrics \
        REMOVE_DUPLICATES=false ASSUME_SORTED=true\
           VALIDATION_STRINGENCY=LENIENT
    samtools view -F 1804 -b ${name}.sorted.marked.bam > ${name}.sorted.nodup.bam
done

samtools merge ${cl}.bam *.sorted.nodup.bam -@ 8

samtools view -b -F 16 ${cl}.bam >${cl}_fwd.bam
samtools view -b -f 16 ${cl}.bam >${cl}_rev.bam
samtools index ${cl}_fwd.bam -@ 8
samtools index ${cl}_rev.bam -@ 8
