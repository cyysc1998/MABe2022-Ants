sudo mkdir /cache
sudo mount -t tmpfs -o size=300G tmpfs /cache
cp /root/data/yuannian_data/jh/MABe2022/data/mouse/frames.tar /cache/
tar -xf /cache/frames.tar -C /cache/
cp /root/data/yuannian_data/jh/MABe2022/data/mouse/*.npy /cache/
cp /root/data/yuannian_data/jh/MABe2022/data/mouse/*.txt /cache/
cp /root/data/yuannian_data/jh/pretrained_models/resnet50-0676ba61.pth /cache/