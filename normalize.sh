cd $1

cwd=$PWD

mkdir reg_standard

flirt -ref reg/standard -in reg/standard -out reg_standard/standard -applyisoxfm 3

flirt -ref reg_standard/standard -in filtered_func_data -out reg_standard/filtered_func_data.nii.gz -applyxfm -init reg/example_func2standard.mat -interp trilinear -datatype float

cd $cwd
