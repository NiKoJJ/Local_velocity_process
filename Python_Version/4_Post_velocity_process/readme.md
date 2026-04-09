

- 2_process_pipline_monthly_yearly

python run_pipeline.py  --gamma-dir /data1/PhD_Work1/autoRIFT_Optical_Local_Pipeline/velocity/velocity_components /data1/PhD_Work1/autoRIFT_S1_Local_Pipeline-v4/velocity/nc_tif --bbox-3031 1022704 -2145536 1150679 -2020695 --modes monthly --vx-error-mode proportional --use-gamma  --gamma-error-mode local_std --local-std-window 5 --temporal-mad  --temporal-mad-k 1.5 --error-threshold 200 --n-workers 3 --io-threads 10 --output-dir /data2/Phd_Work1/ITSLIVE/Finally_result_speed_add_S1_autorift/  --no-itslive --vmax 1000 --spatial-mad-window 15
