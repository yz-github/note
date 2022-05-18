一 安装
yum install -y unzip


二 使用
unzip
1 #直接解压到当前目录下
unzip 1.zip 
2 #通过-d指定解压路径,即解压到当前目录下folder这个文件夹下,如果这个文件夹不存在，可以自动创建
unzip 1.zip -d folder

zip
#zip [选项] 压缩包名 源文件或源目录
zip ana.zip anaconda-ks.cfg
#压缩多个文件
zip test.zip abc abcd
#将aa文件夹下的所有文件压缩成aa.zip
zip -r aa.zip /opt/module/aa/*
