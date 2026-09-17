./infer_e2e_b8_daemon --source /root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test \
        --engine /root/my_FILE/models/yolov8_e2e_dyn.engine --batch 8 \
        --done /root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test_1 \
        --save-img \
        --out /root/my_FILE/infer_images


        
        --workers 3   --consumers 2   --poll-ms 1000 
        --conf 0.45   --iou 0.65
        --no-move-done   --save-txt


        --source /root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/train