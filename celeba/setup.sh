#!/usr/bin/env bash

# Download the faces dataset (1.34GByte)
mkdir data_faces && wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip

# Download the identities of all individuals (3.27 MByte)
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS' -O celeba_ids.txt