
max_pathlens=(3 4 5 6)
max_paths=(2 3 4 5 6 9 10 12 18 20 21 24 26)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "../Results/Output_final_"$max_path"_"$max_pathlen".txt" python main.py $max_pathlen $max_path False False
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "../Results/Output_final_weighted_"$max_path"_"$max_pathlen".txt" python main.py $max_pathlen $max_path False True
	done
done

max_pathlens=(10 12 18 20 21 22 24 26 28 32 38)
max_paths=(2)

for max_pathlen in "${max_pathlens[@]}";
do
	for max_path in "${max_paths[@]}";
	do
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "../Results/Output_final_bon_"$max_path"_"$max_pathlen".txt" python main.py $max_pathlen $max_path True False
		jbsub -q x86_24h -mem 40g -require k80 -cores 1x1+1 -out "../Results/Output_final_bon_weighted_"$max_path"_"$max_pathlen".txt" python main.py $max_pathlen $max_path True True
	done
done
