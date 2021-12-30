
### Get list of sites from a vcf.gz file:
```bash
bcftools query -f '%CHROM\t%POS\n' genomes.vcf.gz > chrBP_list.txt
```

### Get list of samples from a vcf.gz file:
```bash
bcftools query -l genomes.vcf.gz > samples_list.txt
```

### Get vcf.gz file from `genomes.vcf.gz` file with sites present only in file `chrBP_list_EUROFINS.txt`:
```bash
bcftools view -T chrBP_list_EUROFINS.txt genomes.vcf.gz -Oz -o genomes_EUROFINS.vcf.gz
```

### Get vcf.gz file from `genomes.vcf.gz` file with samples present only in file `samples_list_80.txt`:
```bash
bcftools view -S samples_list_80.txt genomes.vcf.gz --threads 1 -Oz -o genomes_80-samples.vcf.gz
```
