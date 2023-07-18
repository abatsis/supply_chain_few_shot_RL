CUDA_VISIBLE_DEVICES="" parallel -j100 --bar "python generate.py >/dev/null 2>/dev/null" ::: {1..1000}


#for var in {1..100}
#do
#  CUDA_VISIBLE_DEVICES="" python generate.py &
#done

