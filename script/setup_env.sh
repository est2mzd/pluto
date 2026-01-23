
# pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# --> Dockerfile　でインストール済み

# pip3 install natten==0.14.6 -f https://shi-labs.com/natten/wheels/cu118/torch2.0.0/index.html
# --> natten は　不要.  [grep -R "natten" -n src/] で確認できる

# pip install -r ./requirements.txt
# --> Dockerfile　でインストール済み