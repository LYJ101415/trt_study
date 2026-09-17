1. 运行C++代码命令（因为是C++代码，不能使用bash,必须进入所在文件夹后运行）：
./infer_e2e_b8_cpp --source /root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test \
                   --engine /root/my_FILE/models/yolov8_e2e_dyn.engine \
                   --batch 8 \
                   --save-img --out /root/my_FILE/infer_images

