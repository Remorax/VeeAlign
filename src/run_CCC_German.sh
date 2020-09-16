# Web directory
# 126 + 108 + 22
max_paths=(1 2 3 4 6 9 13 21 49)
max_pathlens=(1 3 4 5 6 8)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "../Results/Output_final_"$max_path"_"$max_pathlen"_webdir.txt" python main.py $max_pathlen $max_path False False
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "../Results/Output_final_"$max_path"_"$max_pathlen"_webdir_weighted.txt" python main.py $max_pathlen $max_path False True
	done
done

max_paths=(1)
max_pathlens=(1 3 4 5 6 8 9 13 20 21 49)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "../Results/Output_final_"$max_path"_"$max_pathlen"_webdir_bon.txt" python main.py $max_pathlen $max_path True False
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "../Results/Output_final_"$max_path"_"$max_pathlen"_webdir_weighted_bon.txt" python main.py $max_pathlen $max_path True True
	done
done