

for i in *_1.fastq.gz
do
    srrid=$(echo $i | awk -F"/" '{print $NF}' | awk -F"_1.fastq.gz" '{print $1}')
    echo $srrid

    cutadapt -e 0.1 -m 5 -a CTGTCTCTTATA -A CTGTCTCTTATA \
	    -o ${srrid}_1.trim.fastq.gz -p ${srrid}_2.trim.fastq.gz\
	    ${srrid}_1.fastq.gz ${srrid}_2.fastq.gz

    bowtie2 -X 2000 -x /nfs/turbo/umms-drjieliu/proj/bowtie2_index/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta \
	    --mm --threads 10 \
	    -1 ${srrid}_1.trim.fastq.gz \
	    -2 ${srrid}_2.trim.fastq.gz -S ${srrid}.sam &> ${srrid}.txt

    samtools view -Su ${srrid}.sam -@ 8 -o ${srrid}.bam
    rm ${srrid}.sam
    samtools view -F 1804 -f 2 -q 30 -u ${srrid}.bam -o ${srrid}.1.bam -@ 8
    samtools sort -n ${srrid}.1.bam -o ${srrid}.tmp.flit.bam -@ 8
    rm ${srrid}.1.bam

    samtools fixmate -r ${srrid}.tmp.flit.bam ${srrid}.flit.fix.bam -@ 8

    samtools view -F 1804 -f 2 -u ${srrid}.flit.fix.bam -@ 8| \
        samtools sort /dev/stdin -o ${srrid}.flit.bam -@ 8
    rm ${srrid}.tmp.flit.bam
    rm ${srrid}.flit.fix.bam

    module load openjdk/18.0.1.1
    java -jar ../../picard.jar MarkDuplicates I=${srrid}.flit.bam \
        O=${srrid}.flit.tmp.bam METRICS_FILE=${srrid}.metrics \
        VALIDATION_STRINGENCY=LENIENT ASSUME_SORTED=true \
        REMOVE_DUPLICATES=false
    rm ${srrid}.flit.bam
    samtools view -F 1804 -f 2 -b ${srrid}.flit.tmp.bam > ${srrid}.nodup.bam
    rm ${srrid}.flit.tmp.bam
done